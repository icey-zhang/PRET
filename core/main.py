# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import random
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

import numpy as np
import cv2
import openslide
from sklearn.metrics import precision_recall_curve

from modules import inference, load_weak_prompts, execute_tagger, \
        execute_subtyping_tagger, execute_miner


# ====================== collect features and information ======================

def _load_patch_label_mask(v):
    """Load segmentation GT for a slide if present and readable on disk.

    Returns the (H, W) uint8 mask, or None when there is no patch_labels entry
    or the file is missing/unreadable. Classification / screening tasks don't
    need this, so a missing GT must not crash feature_processor.
    """
    if 'patch_labels' not in v:
        return None
    path = v['patch_labels']
    if not os.path.exists(path):
        print('warning: patch_labels file missing, skipping mask: ' + path)
        return None
    img = cv2.imread(path)
    if img is None:
        print('warning: patch_labels file unreadable, skipping mask: ' + path)
        return None
    return img[:, :, 0]


def _load_legacy_npy_slide(in_dir, v, args):
    """Read PRET's per-patch .npy features for a single slide (legacy pipeline).

    Returns (feats list, names list, patch_label list, coords (N,2) int32).
    Coords are patch-grid indices and are reused later to skip per-patch string
    parsing in inference / taggers.
    """
    feats, names, patch_label, coords = [], [], [], []

    mask = _load_patch_label_mask(v)

    in_dir = in_dir if in_dir[-1] != '/' else in_dir[:-1]
    patch_path = in_dir.replace(in_dir.split('/')[-2], 'images')
    ori_dir = sorted([int(_) for _ in os.listdir(patch_path)])[-1]
    patch_path = os.path.join(patch_path, str(ori_dir))

    for f in os.listdir(os.path.join(in_dir, 'x20')):
        name = os.path.join(patch_path, f.replace('.npy', '.jpeg'))
        if os.path.getsize(name) < args.file_min_size:
            continue

        feat = np.load(os.path.join(in_dir, 'x20', f))
        feat = feat / np.linalg.norm(feat, ord=2, axis=0)

        x, y = f.split('.')[0].split('_')
        x, y = int(x), int(y)
        if mask is not None:
            patch_label.append(mask[y, x])

        names.append(name)
        feats.append(feat)
        coords.append((x, y))

    coords = np.asarray(coords, dtype=np.int32) if coords else np.zeros((0, 2), dtype=np.int32)
    return feats, names, patch_label, coords


def _load_trident_h5_slide(h5_path, slide_name, v, args):
    """Read a trident-extracted .h5 feature file for a single slide.

    Expected layout (matches the trident patch feature extractor):
        features  : (N, D) float patch features
        coords    : (N, 2) int patch top-left coordinates in level-0 pixels
        attrs     : 'patch_size_level0' / 'patch_size' on the file or 'coords' dataset

    Returns (feats, names, patch_label) in the same shape as the legacy loader so
    downstream taggers, inference and visualisation keep working unchanged. Patch
    names are synthesised as "<raw_feature_path>/<slide_name>/<x>_<y>.jpeg" so the
    existing coord parsing (basename split by '_') and the "slide_name in patch_name"
    lookups in modules.py still match.
    """
    import h5py

    mask = _load_patch_label_mask(v)

    with h5py.File(h5_path, 'r') as h5:
        features = h5['features'][:]
        coords_pix = h5['coords'][:]

        # discover the patch size in level-0 pixels (used to map coords -> grid index)
        attrs = {}
        attrs.update(dict(h5.attrs))
        if 'coords' in h5:
            attrs.update(dict(h5['coords'].attrs))
        patch_size_l0 = int(attrs.get('patch_size_level0',
                                      attrs.get('patch_size',
                                                args.patch_scale)))
        if patch_size_l0 <= 0:
            patch_size_l0 = args.patch_scale

    # Vectorised: pixel coords -> patch-grid indices, L2-normalise rows.
    coords = (np.asarray(coords_pix[:, :2], dtype=np.int64) // patch_size_l0).astype(np.int32)
    feats_arr = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(feats_arr, ord=2, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats_arr = feats_arr / norms

    # patch_label sampled from mask grid (vectorised gather with bounds check)
    patch_label = []
    if mask is not None:
        ys = np.clip(coords[:, 1], 0, mask.shape[0] - 1)
        xs = np.clip(coords[:, 0], 0, mask.shape[1] - 1)
        oob = (coords[:, 1] < 0) | (coords[:, 1] >= mask.shape[0]) | \
              (coords[:, 0] < 0) | (coords[:, 0] >= mask.shape[1])
        sampled = mask[ys, xs]
        sampled[oob] = 0
        patch_label = list(sampled)

    # synthesise patch names that still encode <x>_<y> for any caller that
    # falls back to filename parsing
    patch_path = os.path.join(args.raw_feature_path, slide_name)
    names = ['%s/%d_%d.jpeg' % (patch_path, int(coords[i, 0]), int(coords[i, 1]))
             for i in range(coords.shape[0])]
    feats = list(feats_arr)

    return feats, names, patch_label, coords


def _dataset_match(args, hint):
    """True if the dataset corresponds to `hint` (e.g. 'TCGA' / 'CAMELYON' / 'LN').

    Looks at --wsi_path, --dataset_info path, and the first slide name in the
    dataset_info JSON. This preserves dataset-specific behaviour (TCGA
    same-patient filtering, CAMELYON binary sampling, ...) even when
    --wsi_path is left empty because features were extracted by trident.
    """
    if hint in (args.wsi_path or ''):
        return True
    if hint in (args.dataset_info or ''):
        return True
    try:
        with open(args.dataset_info) as fh:
            d = json.load(fh)
        for k in d:
            return hint in k
    except Exception:
        pass
    return False


def _resolve_wsi_suffix(wsi_path):
    """Return the WSI file extension found in wsi_path, or '' if unavailable.

    Pure-classification runs on trident-extracted h5 features don't need the raw
    WSIs at all. When wsi_path is missing/empty we fall back to '', and the
    downstream os.path.exists guards on the joined path keep openslide from
    being called (size becomes None -> heatmap branch is skipped).
    """
    try:
        files = [f for f in os.listdir(wsi_path) if not f.startswith('.')]
    except (FileNotFoundError, NotADirectoryError, TypeError):
        return ''
    return files[0].split('.')[-1] if files else ''


def feature_processor(args):
    print('start feature processing ...')
    dataset_info = json.load(open(args.dataset_info))
    os.makedirs(args.dump_features, exist_ok=True)

    n_done, n_skip_existing, n_skip_missing, n_skip_empty = 0, 0, 0, 0
    missing_examples = []

    for k, v in dataset_info.items():
        if os.path.exists(os.path.join(args.dump_features, k + '.npy')):
            n_skip_existing += 1
            continue

        wsi_label = v['wsi_label']

        # auto-detect feature source: trident-extracted .h5 (one file per slide)
        # takes priority; otherwise fall back to PRET's legacy per-patch .npy dir.
        h5_path = os.path.join(args.raw_feature_path, k + '.h5')
        legacy_dir = os.path.join(args.raw_feature_path, k + '_files')

        if os.path.exists(h5_path):
            feats, names, patch_label, coords = _load_trident_h5_slide(h5_path, k, v, args)
        elif os.path.isdir(legacy_dir):
            feats, names, patch_label, coords = _load_legacy_npy_slide(legacy_dir, v, args)
        else:
            n_skip_missing += 1
            if len(missing_examples) < 5:
                missing_examples.append(k)
            continue

        if len(names) == 0:
            n_skip_empty += 1
            continue

        # save patch features, name, patch_labels, coords and wsi_labels for eval.
        # 'coords' (N,2) int32 lets downstream code skip per-patch string parsing.
        info = {'features': np.stack(feats, 0), 'patch_names': names,
                'patch_labels': np.array(patch_label),
                'coords': np.asarray(coords, dtype=np.int32),
                'wsi_label': wsi_label}
        np.save(os.path.join(args.dump_features, k + '.npy'), info)
        n_done += 1

    if n_skip_missing:
        print('warning: %d slides have no extracted features (missing .h5/_files dir under %s); '
              'first few: %s' % (n_skip_missing, args.raw_feature_path, missing_examples))
    if n_skip_empty:
        print('warning: %d slides skipped with 0 valid patches' % n_skip_empty)
    print('finish feature processing: %d new, %d already cached, %d missing, %d empty (total %d)'
          % (n_done, n_skip_existing, n_skip_missing, n_skip_empty, len(dataset_info)))


# ====================== some util functions ======================

# Support (example) slides are cached unconditionally for the lifetime of the
# process. They are reused across runs / classes / shot counts and there are
# at most O(args.example_num) of them, so the memory footprint is bounded by
# the chosen support set, not the dataset size. Query slides are NOT cached;
# they are streamed through a DataLoader (see _QuerySlideDataset below) so
# multiple worker processes overlap pickle decoding with GPU inference.
_SUPPORT_CACHE = {}


def _load_support_features(args, name):
    cached = _SUPPORT_CACHE.get(name)
    if cached is not None:
        return cached
    data = np.load(os.path.join(args.dump_features, name + '.npy'),
                   allow_pickle=True).item()
    _SUPPORT_CACHE[name] = data
    return data


def _get_coords(slide_dict):
    """Return (N, 2) int32 patch-grid coords for a slide dict.

    New dicts written by feature_processor already have a 'coords' key.
    For older dicts (or hand-built ones) we lazily parse the per-patch
    filenames once and memoise back into the dict.
    """
    coords = slide_dict.get('coords')
    if coords is not None:
        return coords
    pn = slide_dict.get('patch_names', [])
    if len(pn) == 0:
        coords = np.zeros((0, 2), dtype=np.int32)
    else:
        out = np.empty((len(pn), 2), dtype=np.int32)
        for i, p in enumerate(pn):
            x, y = p.split('/')[-1].split('.')[0].split('_')
            out[i, 0] = int(x); out[i, 1] = int(y)
        coords = out
    slide_dict['coords'] = coords
    return coords


class _QuerySlideDataset(torch.utils.data.Dataset):
    """Yields one query slide per __getitem__.

    Workers run in separate processes so the pickle decode of each
    <slide>.npy happens in parallel with the GPU inference of the
    previous slide. Returned tensors are CPU; the main process moves
    them to GPU just before inference.
    """
    def __init__(self, dump_features, names):
        self.dump_features = dump_features
        self.names = list(names)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        data = np.load(os.path.join(self.dump_features, name + '.npy'),
                       allow_pickle=True).item()
        return {
            'name': name,
            'features': torch.from_numpy(np.ascontiguousarray(data['features'])),
            'patch_names': data['patch_names'],
            'patch_labels': data.get('patch_labels'),
            'coords': _get_coords(data),
            'wsi_label': data['wsi_label'],
        }


def _query_loader(args, names):
    nw = max(0, int(getattr(args, 'num_workers', 0) or 0))
    ds = _QuerySlideDataset(args.dump_features, names)
    return torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=nw,
        collate_fn=lambda batch: batch[0],   # we always run one slide at a time
        pin_memory=False,
        persistent_workers=False,            # evaluate() may be called per-shot
    )


def _threshold_sweep(val_preds, val_labels, thresholds):
    """Vectorised replacement for the per-threshold list comprehension.

    val_preds:   1-D float (N,)
    val_labels:  1-D int/float 0/1 (N,)
    thresholds:  1-D float (T,)
    Returns accs as 1-D float (T,) with the same semantics as the original
    [((val_preds > t).float() == val_labels).sum() / N for t in thresholds].
    """
    if hasattr(val_preds, 'numpy'):
        val_preds = val_preds.numpy()
    if hasattr(val_labels, 'numpy'):
        val_labels = val_labels.numpy()
    val_labels_bool = val_labels.astype(bool)
    # broadcast: (T, N) bool grid
    pred_grid = val_preds[None, :] > thresholds[:, None]
    return (pred_grid == val_labels_bool[None, :]).mean(axis=1)


def macro_value(l, n):
    out = []
    for i in range(len(l) // n):
        v = sum(l[i * n: i * n + n]) / n
        out.append(v)
    return out


def get_example_names_at_same_num(all_names, dataset_info, example_num, check_num=False):
    record = {}
    for n in all_names:
        lb = dataset_info[n]['wsi_label']
        if lb not in record:
            record[lb] = []
        record[lb].append(n)

    names = []
    for k, v in record.items():
        if check_num == True and len(v) < (example_num):
            print('exist! insufficient samples. ' + str(k))
            sys.exit(0)
        names.extend(v[:example_num])

    return names


def check_different_patient(example_names, query_candidates, mode='TCGA'):
    out = []
    if mode == 'TCGA':
        for q in query_candidates:
            inside = False
            for g in example_names:
                if mode == 'TCGA':
                    if q[:12] == g[:12]:
                        inside = True
            if not inside:
                out.append(q)

    return out


# post processing via gussain blur
class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(GaussianBlur, self).__init__()
        kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-kernel_size//2)**2 + (y-kernel_size//2)**2)/(2*sigma**2)),
        (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        kernel = np.reshape(kernel, (1, 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(torch.from_numpy(kernel).float(), requires_grad=False).cuda()

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.weight.shape[-1]//2)


# ====================== evaluation for multiple tasks, prompts, and shots ======================

def evaluate(args, val_only=False):
    auc_list, f1_list, acc_list, example_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    info_str = open(args.dataset_info).read()
    dataset_info = json.load(open(args.dataset_info))
    all_names = dataset_info.keys()

    # drop slides whose features failed to extract (matches evaluate_baseline)
    available, missing = [], []
    for _ in all_names:
        if os.path.exists(os.path.join(args.dump_features, _ + '.npy')):
            available.append(_)
        else:
            missing.append(_)
    if missing:
        print('warning: %d/%d slides have no collected features and will be skipped (e.g. %s)'
              % (len(missing), len(missing) + len(available), missing[0]))
    all_names = available

    records = {}
    txt_rec = []
    # ====================== repeat experimets n=args.runs ======================

    for i in range(args.runs):
        records['repeat_' + str(i)] = {}
        
        # ====================== data split ======================

        labeled_names, neg_names, test_names, rest_names = [], [], [], []

        for n in all_names:
            # splitdata, if there is fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)

            else:
                # pick pos from labeled wsi
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']
                    
                    # prompt samplinging (camelyon only)
                    if args.c == 1 and _dataset_match(args, 'CAMELYON'):
                        if pn >= 1000 and pn < 3000:
                            labeled_names.append(n)
                    
                    else:
                        labeled_names.append(n)
                
                if args.prompt_type == 'slideLabel':
                    # add neg and pos for subtyping (no labeled wsis)
                    if args.c > 1 and 'pos_patch_num' not in info_str:
                        labeled_names.append(n)
                
                    # add some neg for slideLabel binary cls
                    if args.c == 1 and dataset_info[n]['wsi_label'] == 0:
                        labeled_names.append(n)
                
                # record neg names to exclude from seg val /test
                if dataset_info[n]['wsi_label'] == 0:
                    neg_names.append(n)

        # shuffle example till each run is different
        while True:
            random.shuffle(labeled_names)

            # randomly select "args.example_num" examples for each class
            # note: for binary tasks 'slideLabel' use N // 2 pos and N // 2 neg
            if args.c > 1 or args.prompt_type == 'slideLabel':
                example_i = get_example_names_at_same_num(labeled_names, dataset_info, args.example_num, args.c > 1)

            # randomly select "args.example_num" positive examples for binary tasks
            else:
                example_i = labeled_names[:args.example_num]

            # avoid repeat example
            example_i.sort()
            if example_i not in example_list:
                example_list.append(example_i)
                example_names = example_i
                break

        # split val set out of example and test set
        example_set = set(example_names)
        neg_set = set(neg_names)
        for n in all_names:
            if n not in example_set and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)

        if args.seg:
            rest_names = []
            for ln in labeled_names:
                if ln not in example_set and ln not in neg_set:
                    rest_names.append(ln)

        # avoid same patients in different split, in-house data is cleaned
        if _dataset_match(args, 'TCGA'):
            rest_names = check_different_patient(example_names, rest_names, 'TCGA')

        random.shuffle(rest_names)
        val_num = args.val_num if args.val_ratio < 0 else int(len(rest_names) * args.val_ratio)
        val_names = rest_names[:val_num]

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            if args.val_ratio < 0:
                test_names = rest_names[-args.test_num:]
            else:
                test_names = rest_names[val_num:]
            if len(val_names) + len(test_names) > len(rest_names):
                print('wrong split size !!!')
        else: # take partial test slides for tcga cross races
            random.shuffle(test_names)
            if args.test_num > 0:
                test_names = test_names[:args.test_num]
        
        records['repeat_' + str(i)]['split'] = {'example_names': example_names, 'val_names': val_names, 'test_names': test_names}

        # ====================== run for each class ======================

        # for subtyping, use different example for each cls and apply marco metics, other tasks have one class
        for cls in range(1, args.c + 1):

            # ====================== process example and prompts ======================

            # load example
            example_feats, example_patch_names, example_labels = [], [], []
            example_slide_coords = []   # one (n_j, 2) int array per WSI
            example_slide_offsets = [0] # cumulative patch counts -> (W+1,)
            for n in example_names:
                example_n = _load_support_features(args, n)
                example_patch_names = example_patch_names + example_n['patch_names']
                example_feats.append(example_n['features'])
                example_slide_coords.append(_get_coords(example_n))
                example_slide_offsets.append(example_slide_offsets[-1] + example_n['features'].shape[0])

                # empty patch label for image label or sparse label where there is no offline gt
                if args.prompt_type == 'mask':
                    pl = example_n['patch_labels'].copy()  # mutated below; cache stays clean

                    # binary use 0 normal, 1 tumor, while subtyping use 0 other cls, 1 this cls, 255 normal
                    if args.c > 1:
                        pl[pl == 0] = 255
                        if example_n['wsi_label'] != cls:
                            pl[pl == 1] = 0
                        else:
                            pl[pl == 1] = 1
                    
                else:
                    pl = np.zeros(example_n['features'].shape[0]) - 1
                
                # load weak prompts
                # slideLabel + subtyping is uniqe in pseudo label generation
                if args.prompt_type == "slideLabel" and args.c > 1:
                    if example_n['wsi_label'] != cls:
                        pl[:] = 0
                    else:
                        pl[:] = 1

                # for box, RoughMask and binary + slideLabel, -1 is uncertain pos, 0 is normal
                elif args.prompt_type != 'mask' :
                    pl = load_weak_prompts(n, example_n['wsi_label'], args.wsi_path, pl, \
                        example_n['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale)
                    
                    #  record wsi label for each patch for later label convert
                    if args.c > 1:
                        pl[pl == 0] = 255
                        pl[pl == -1] = 1 if example_n['wsi_label'] == cls else 0

                example_labels.append(pl)
            
            example_feats = torch.tensor(np.concatenate(example_feats, 0)).cuda()
            example_labels = torch.tensor(np.concatenate(example_labels, 0)).cuda().long()

            if args.dump_pseudo != '':
                vis_info = {'wsi_dir': args.wsi_path, 'vis_dir': os.path.join(args.dump_pseudo, 'vis') + str(args.example_num) + '/' + str(i) + '/' + str(cls), \
                        'mask_dir': os.path.join(args.dump_pseudo, 'pseudo') + str(args.example_num) + '/' + str(i) + '/' + str(cls)}

                split_dir = os.path.join(args.dump_pseudo, 'split') + str(args.example_num)
                split = {'example_names': example_names, 'test_names': test_names, 'val_names': val_names}
                os.makedirs(split_dir, exist_ok=True)
                open(os.path.join(split_dir, str(i) + '.json'), 'w').write(json.dumps(split, indent=4))

            else:
                vis_info = None

            # ====================== apply in-context tagger ======================

            # assign in-context tags for weak prompts (binary tasks: 1 pos, 0 neg, -1 unknown)
            if args.prompt_type != 'mask' and args.c == 1:
                example_labels = execute_tagger(example_feats, example_labels, example_patch_names, example_names, \
                    vis_info=vis_info, uncertain=args.ignore, topk=args.topk,
                    slide_offsets=example_slide_offsets, slide_coords=example_slide_coords)

            # assign in-context tags for subtyping from slideLabel (255 normal, 254 uncertain, 1 this class, 0 other classes)
            if args.prompt_type == 'slideLabel' and args.c > 1:
                example_labels = execute_subtyping_tagger(example_feats, example_labels, example_patch_names, \
                    example_names, vis_info=vis_info, uncertain=args.ignore, topk=args.topk,
                    slide_offsets=example_slide_offsets, slide_coords=example_slide_coords)

            # subtyping + box / roughMask. Need to process "execute_tagger" twice.
            # Once for shared bg and this class, another for shared bg and other classes
            if args.prompt_type != 'slideLabel' and args.c > 1:

                if args.prompt_type != 'mask':
                    example_labels_this = example_labels.clone()
                    example_labels_this[example_labels_this == 0] = 254 # ignore other fg
                    example_labels_this[example_labels_this == 255] = 0 # subtyping bg label to binary neg label
                    example_labels_this[example_labels_this == 1] = -1  # this class to undertain to relabel
                    example_labels_this = execute_tagger(example_feats, example_labels_this, example_patch_names, example_names, \
                        vis_info=vis_info, uncertain=args.ignore, topk=args.topk,
                        slide_offsets=example_slide_offsets, slide_coords=example_slide_coords)

                    vis_info = None
                    example_labels_others = example_labels.clone()
                    example_labels_others[example_labels_others == 1] = 254 # ignore this fg
                    example_labels_others[example_labels_others == 0] = -1  # other class to undertain to relabel
                    example_labels_others[example_labels_others == 255] = 0 # subtyping bg label to binary neg label
                    example_labels_others = execute_tagger(example_feats, example_labels_others, example_patch_names, example_names, \
                        vis_info=vis_info, uncertain=args.ignore, topk=args.topk,
                        slide_offsets=example_slide_offsets, slide_coords=example_slide_coords)

                    example_labels[:] = 255 # default bg
                    example_labels[example_labels_this == 1] = 1     # this class
                    example_labels[example_labels_others == 1] = 0   # other class
                    example_labels[example_labels_this == -1] = 254  # ignore in the last
                    example_labels[example_labels_others == -1] = 254# ignore in the last
                    if (example_labels == 255).sum() == 0:
                        example_labels[example_labels_others == 0] = 255
                        example_labels[example_labels_this == 0] = 255

            # ====================== predict for test slides (queries)======================

            # predict for test slides, name a test slide as query to avoid confusion with test set
            val_preds, test_preds, val_labels, test_labels = [], [], [], []
            wsi_suffix = _resolve_wsi_suffix(args.wsi_path)
            all_query_names = val_names if val_only else val_names + test_names
            # O(1) membership lookups in the per-slide loop below
            val_set = set(val_names)
            test_set = set(test_names)
            sm = GaussianBlur(7, 3) if args.seg else None  # build once per class
            # parallel pickle decode: workers stream the next query slides
            # while the main process is busy doing GPU inference on the current one
            for query_n in _query_loader(args, all_query_names):
                n = query_n['name']
                query_feats = query_n['features'].cuda(non_blocking=True)
                query_patch_names = query_n['patch_names']
                label = query_n['wsi_label']
                if args.c > 1:
                    label = int(label == cls)

                # ====================== discriminative instance miner for subtyping ======================

                # use fg patches for subtyping
                #if args.c > 1 and not args.seg and args.vis_path == '': # vis wo fg
                if args.c > 1 and not args.seg:
                    query_feats, query_patch_names = execute_miner(example_feats[example_labels == 255], \
                        query_feats, query_patch_names, uncertain=args.ignore_query)

                # ====================== inference, including classifier, aggregator, post processer ======================

                wsi_path = os.path.join(args.wsi_path, n + '.' + wsi_suffix)
                if os.path.exists(wsi_path):
                    wsi = openslide.OpenSlide(wsi_path)
                    size = (wsi.level_dimensions[0][1] // args.patch_scale, wsi.level_dimensions[0][0] // args.patch_scale)
                else:
                    size = None
                vis_info = None

                wsi_pred, patch_pred, patch_pred_list = inference(args, example_feats, example_labels, example_patch_names, \
                    query_feats, query_patch_names, size, args.top_instance, vis_info, smooth=sm,
                    query_coords=query_n['coords'])

                if patch_pred != None and args.vis_path != '' and n in test_set:
                    os.makedirs(args.vis_path, exist_ok=True)
                    np.save(os.path.join(args.vis_path, n + '_' + str(cls) + '.npy'), patch_pred.cpu().numpy())

                if args.seg:
                    pred = torch.tensor(patch_pred_list)
                    label = torch.tensor(query_n['patch_labels'])
                else:
                    pred = torch.tensor([wsi_pred])
                    label = torch.tensor([label])

                if n in val_set:
                    val_preds.append(pred)
                    val_labels.append(label)
                else:
                    test_preds.append(pred)
                    test_labels.append(label)

            # ====================== process validation set and assign label ======================

            # Evaluate on the val set to make sure qualified results for application
            # Val set also guidances to select prediction threshod, f1 for seg. acc for others
            val_preds = torch.cat(val_preds).cpu()
            val_labels = torch.cat(val_labels)
            val_auc = aucroc(val_preds, val_labels).item()
            if not val_only:
                test_preds = torch.cat(test_preds).cpu()
                test_labels = torch.cat(test_labels)
            
            precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
            accs = _threshold_sweep(val_preds, val_labels, thresholds)
            if args.seg:
                f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_acc_score = accs[best_f1_score_index]
                thresh = thresholds[best_f1_score_index]
            else:
                best_acc_score = np.max(accs[np.isfinite(accs)])
                best_acc_score_index = np.argmax(accs[np.isfinite(accs)])
                thresh = thresholds[best_acc_score_index]

            if val_only:
                preds = val_preds
                thresh_preds = (val_preds > thresh).float()
                labels = val_labels
            else:
                preds = test_preds
                thresh_preds = (test_preds > thresh).float()
                labels = test_labels

            acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
            rec = ((thresh_preds * labels).sum() / labels.sum()).cpu().item()
            pre = ((thresh_preds * labels).sum() / thresh_preds.sum()).cpu().item()
            auc = aucroc(preds, labels).item()
            if (rec + pre) != 0:
                f1 = rec * pre * 2 / (rec + pre)
            else:
                f1 = 0
            auc_list.append(auc)
            f1_list.append(f1)
            acc_list.append(acc)
            if not val_only:
                s = 'class:' + str(cls) + ' val auc:' + str(round(val_auc, 4)) + ', test auc:' + str(round(auc, 4)) + \
                    ', val acc: ' + str(round(best_acc_score, 4)) + ', test f1: ' + str(round(f1, 4)) + \
                    ', test acc: ' + str(round(acc, 4))
                print(s)
                txt_rec.append(s)
                records['repeat_' + str(i)]['results_cls' + str(cls)] = {'val_auc': round(val_auc, 4), 'test_auc': round(auc, 4), \
                        'val_acc': round(best_acc_score, 4), 'test_f1': round(f1, 4), 'test_acc': round(acc, 4)}
                records['repeat_' + str(i)]['pred_cls' + str(cls)] = {'labels': labels.cpu().tolist(), \
                        'logits': preds.cpu().tolist(), 'preds': thresh_preds.cpu().tolist()}

        del example_feats, query_feats
        torch.cuda.empty_cache()

    # ====================== count and record results ======================

    auc_mean = np.array(auc_list).mean()
    macro_auc = macro_value(auc_list, args.c)
    auc_std = np.array(macro_auc).std()
    f1_mean = np.array(f1_list).mean()
    macro_f1 = macro_value(f1_list, args.c)
    f1_std = np.array(macro_f1).std()
    acc_mean = np.array(acc_list).mean()
    macro_acc = macro_value(acc_list, args.c)
    acc_std = np.array(macro_acc).std()
    s = 'auc mean: ' + str(round(auc_mean, 4)) + ', auc std: ' + str(round(auc_std, 4)) + \
        ', f1 mean: ' + str(round(f1_mean, 4)) + ', f1 std: ' + str(round(f1_std, 4)) + \
        ', acc mean: ' + str(round(acc_mean, 4)) + ', acc std: ' + str(round(acc_std, 4))
    print(s)
    txt_rec.append(s)

    records['mean'] = {'auc_mean': round(auc_mean, 4), 'auc_std': round(auc_std, 4), 'auc_values': macro_auc, \
            'f1_mean': round(f1_mean, 4), 'f1_std': round(f1_std, 4), 'f1_values': macro_f1, \
            'acc_mean': round(acc_mean, 4), 'acc_std': round(acc_std, 4), 'acc_values': macro_acc}
    records['text_records'] = txt_rec
     
    return round(auc_mean, 4), records


# ====================== evaluation for baseline methods ======================

def evaluate_baseline(args, mode):
    auc_list, f1_list, acc_list, example_list = [], [], [], []
    aucroc = torchmetrics.AUROC(task='binary', num_classes=1)
    info_str = open(args.dataset_info).read()
    dataset_info = json.load(open(args.dataset_info))
    all_names = dataset_info.keys()

    # skip invalid wsis
    temp = []
    for _ in all_names:
        if os.path.exists(os.path.join(args.dump_features, _ + '.npy')):
            temp.append(_)
    all_names = temp

    # ====================== run for each class ======================

    records = {}
    txt_rec = []
    for i in range(args.runs):
        records['repeat_' + str(i)] = {}

        # ====================== data split ======================

        # data split
        labeled_names, neg_names, test_names, rest_names = [], [], [], []

        for n in all_names:
            # splitdata, if there is fixed test set
            if dataset_info[n]['fixed_test_set']:
                test_names.append(n)

            else:
                # pick pos from labeled wsi
                if 'pos_patch_num' in dataset_info[n]:
                    pn = dataset_info[n]['pos_patch_num']

                    # prompt samplinging (camelyon only)
                    if args.c == 1 and _dataset_match(args, 'CAMELYON'):
                        if pn >= 1000 and pn < 3000:
                            labeled_names.append(n)

                    else:
                        labeled_names.append(n)

                if args.prompt_type == 'slideLabel':
                    # add neg and pos for subtyping (no labeled wsis)
                    if args.c > 1 and 'pos_patch_num' not in info_str:
                        labeled_names.append(n)

                    # add some neg for slideLabel binary tasks
                    if args.c == 1 and dataset_info[n]['wsi_label'] == 0:
                        labeled_names.append(n)

                # record neg names to exclude from seg val /test
                if dataset_info[n]['wsi_label'] == 0:
                    neg_names.append(n)

        # shuffle example till each run is different
        while True:
            random.shuffle(labeled_names)

            # randomly select "args.example_num" examples for each class
            # note: for binary tasks 'slideLabel' use N // 2 pos and N // 2 neg
            if args.c > 1 or args.prompt_type == 'slideLabel':
                example_i = get_example_names_at_same_num(labeled_names, dataset_info, args.example_num, args.c > 1)

            # randomly select "args.example_num" positive examples for binary tasks
            else:
                example_i = labeled_names[:args.example_num]

            # avoid repeat example
            example_i.sort()
            if example_i not in example_list:
                example_list.append(example_i)
                example_names = example_i
                break

        # split val set out of example and test set
        example_set = set(example_names)
        neg_set = set(neg_names)
        for n in all_names:
            if n not in example_set and dataset_info[n]['fixed_test_set'] == False:
                rest_names.append(n)

        if args.seg:
            rest_names = []
            for ln in labeled_names:
                if ln not in example_set and ln not in neg_set:
                    rest_names.append(ln)

        if _dataset_match(args, 'TCGA'):
            rest_names = check_different_patient(example_names, rest_names, 'TCGA')
        if _dataset_match(args, 'LN'):
            rest_names = check_different_patient(example_names, rest_names, 'LN')

        random.shuffle(rest_names)
        val_num = args.val_num if args.val_ratio < 0 else int(len(rest_names) * args.val_ratio)
        val_names = rest_names[:val_num]

        # split test set by ratio, if no fixed test set
        if len(test_names) == 0:
            if args.val_ratio < 0:
                test_names = rest_names[-args.test_num:]
            else:
                test_names = rest_names[val_num:]
            if len(val_names) + len(test_names) > len(rest_names):
                print('wrong split size !!!')
        else: # take partial test slides for tcga cross races
            random.shuffle(test_names)
            if args.test_num > 0:
                test_names = test_names[:args.test_num]

        records['repeat_' + str(i)]['split'] = {'example_names': example_names, 'val_names': val_names, 'test_names': test_names}

        # ====================== run for each class ======================

        # for subtyping, use different example for each cls and apply marco metics
        for cls in range(1, args.c + 1):

            # load example
            example_feats, example_labels = [], []
            pos_feats, neg_feats = [], []

            # ====================== process example ======================

            for n in example_names:
                example_n = _load_support_features(args, n)

                # empty patch label for image label or sparse label where there is no offline gt
                if args.prompt_type == 'mask':
                    pl = example_n['patch_labels'].copy()  # mutated below; cache stays clean

                    # binary use 0 normal, 1 tumor, while subtyping use 0 other cls, 1 this cls, 255 normal
                    if args.c > 1:
                        pl[pl == 0] = 255
                        if example_n['wsi_label'] != cls:
                            pl[pl == 1] = 0
                        else:
                            pl[pl == 1] = 1

                else:
                    pl = np.zeros(example_n['features'].shape[0]) - 1

                # load sparse label
                # slideLabel + subtyping is uniqe in pseudo label generation
                if args.prompt_type == "slideLabel" and args.c > 1:
                    if example_n['wsi_label'] != cls:
                        pl[:] = 0
                    else:
                        pl[:] = 1

                # for box, RoughMask and binary + slideLabel, -1 is uncertain pos, 0 is normal
                elif args.prompt_type != 'mask' :
                    pl = load_weak_prompts(n, example_n['wsi_label'], args.wsi_path, pl, \
                        example_n['patch_names'], args.prompt_path, args.prompt_type, side=args.patch_scale)

                    #  record wsi label for each patch for later label convert
                    if args.c > 1:
                        pl[pl == 0] = 255
                        pl[pl == -1] = 1 if example_n['wsi_label'] == cls else 0
                
                if 'prototype' in mode:
                    pos_feats.append(example_n['features'][(pl != 0) * (pl != 255)])
                    neg_feats.append(example_n['features'][pl == 0])

                if 'knn' in mode:

                    if args.prompt_type != 'slideLabel':
                        feat_fg = example_n['features'][(pl != 0) * (pl != 255)]
                        if feat_fg.shape[0] != 0:
                            if 'mean' in mode:
                                example_feats.append(feat_fg.mean(0, keepdims=True))
                            elif 'max' in mode:
                                example_feats.append(feat_fg.max(0, keepdims=True))
                            example_labels.append(1)
                        
                        feat_bg = example_n['features'][pl == 0]
                        if feat_bg.shape[0] != 0:
                            if 'mean' in mode:
                                example_feats.append(feat_bg.mean(0, keepdims=True))
                            elif 'max' in mode:
                                example_feats.append(feat_bg.max(0, keepdims=True))
                            example_labels.append(0)
                    else:
                        feat = example_n['features']
                        if 'mean' in mode:
                            example_feats.append(feat.mean(0, keepdims=True))
                        elif 'max' in mode:
                            example_feats.append(feat.max(0, keepdims=True))
                        example_labels.append(1 if example_n['wsi_label'] == cls else 0)

            if 'prototype' in mode:
                example_labels = [1, 0]
                pos_feats = np.concatenate(pos_feats, 0)
                neg_feats = np.concatenate(neg_feats, 0)

                if 'simple_shot' in mode:
                    mean_feat = np.concatenate([pos_feats, neg_feats], 0).mean(0)
                    pos_feats -= mean_feat
                    pos_feats = pos_feats.mean(0, keepdims=True)
                    pos_feats = pos_feats / np.linalg.norm(pos_feats, 2, 1, keepdims=True)
                    neg_feats -= mean_feat
                    neg_feats = neg_feats.mean(0, keepdims=True)
                    neg_feats = neg_feats / np.linalg.norm(neg_feats, 2, 1, keepdims=True)
                    example_feats = [pos_feats, neg_feats]
                else:
                    example_feats = [pos_feats.mean(0, keepdims=True), neg_feats.mean(0, keepdims=True)]

            example_feats = torch.tensor(np.concatenate(example_feats, 0)).cuda()
            example_labels = torch.tensor(example_labels).cuda()

            # ====================== inference for test slides ======================

            # predict query
            val_preds, test_preds, val_labels, test_labels = [], [], [], []
            all_query_names = val_names + test_names
            val_set = set(val_names)
            test_set = set(test_names)
            for query_n in _query_loader(args, all_query_names):
                n = query_n['name']
                query_feats = query_n['features'].cuda(non_blocking=True)
                query_patch_names = query_n['patch_names']
                if args.c > 1:
                    label = query_n['wsi_label'] == cls
                else:
                    label = query_n['wsi_label']

                if 'prototype' in mode:
                    if 'simple_shot' in mode:
                        query_feats -= torch.tensor(mean_feat).cuda()
                        query_feats = query_feats / torch.linalg.norm(query_feats, 2, 1, keepdims=True)

                    topk = min(args.top_instance, query_feats.shape[0])
                    prob = query_feats @ example_feats[0]
                    wsi_pred = prob.topk(topk)[0].mean()

                    if args.vis_path != '' or args.seg:
                        wsi_suffix = _resolve_wsi_suffix(args.wsi_path)
                        wsi_path = os.path.join(args.wsi_path, n + '.' + wsi_suffix)
                        wsi = openslide.OpenSlide(wsi_path)
                        size = (wsi.level_dimensions[0][1] // args.patch_scale, wsi.level_dimensions[0][0] // args.patch_scale)
                        H, W = size
                        patch_pred = torch.full((H, W), 255.0, device=prob.device)

                        # batched coord scatter (replaces per-patch parse + try/except)
                        coords_q = _get_coords(query_n)
                        x_t = torch.from_numpy(np.asarray(coords_q[:, 0], dtype=np.int64)).to(prob.device)
                        y_t = torch.from_numpy(np.asarray(coords_q[:, 1], dtype=np.int64)).to(prob.device)
                        in_bounds = (x_t >= 0) & (x_t < W) & (y_t >= 0) & (y_t < H)
                        flat_full = (y_t.clamp_(0, H - 1) * W + x_t.clamp_(0, W - 1)).long()
                        if in_bounds.any():
                            patch_pred.view(-1).scatter_(0, flat_full[in_bounds], prob[in_bounds])
                        idx_in_map = flat_full

                        if args.vis_path != '' and n in test_set:
                            os.makedirs(args.vis_path, exist_ok=True)
                            np.save(os.path.join(args.vis_path, n + '_' + str(cls) + '.npy'), patch_pred.cpu().numpy())

                        if args.seg:
                            smooth = GaussianBlur(7, 3)
                            fg = patch_pred != 255
                            bg = fg == False
                            smooth_pred = patch_pred.clone()
                            smooth_pred[bg] = smooth_pred[fg].mean() # replace 255 to mean value before smoothing
                            smooth_pred = smooth(smooth_pred.reshape(1, 1, smooth_pred.shape[0], smooth_pred.shape[1]))[0,0]
                            patch_pred[fg] = smooth_pred[fg]
                            patch_pred_list = patch_pred.reshape(-1)[idx_in_map]

                elif 'knn' in mode:
                    if 'mean' in mode:
                        query_feats = query_feats.mean(0)
                    elif 'max' in mode:
                        query_feats = query_feats.max(0)[0]
                    else:
                        print('false eval mode')
                   
                    pos_example_feats, neg_example_feats = example_feats[example_labels == 1], example_feats[example_labels == 0]
                    wsi_pred = (pos_example_feats @ query_feats).topk(min(5, pos_example_feats.shape[0]))[0].mean() - \
                            (neg_example_feats @ query_feats).topk(min(5, neg_example_feats.shape[0]))[0].mean()
                    
                else:
                    print('false eval mode')

                if args.seg:
                    pred = torch.tensor(patch_pred_list)
                    label = torch.tensor(query_n['patch_labels'])
                else:
                    pred = torch.tensor([wsi_pred])
                    label = torch.tensor([label])

                if n in val_set:
                    val_preds.append(pred)
                    val_labels.append(label)
                else:
                    test_preds.append(pred)
                    test_labels.append(label)

            # ====================== process validation set and assign label ======================

            # search a threshold to predict label on val set for fair comparisions
            val_preds = torch.cat(val_preds).cpu()
            val_labels = torch.cat(val_labels)
            val_auc = aucroc(val_preds, val_labels).item()
            test_preds = torch.cat(test_preds).cpu()
            test_labels = torch.cat(test_labels)
            
            precisions, recalls, thresholds = precision_recall_curve(val_labels.numpy(), val_preds.numpy())
            accs = _threshold_sweep(val_preds, val_labels, thresholds)
            if args.seg:
                f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
                best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
                best_acc_score = accs[best_f1_score_index]
                thresh = thresholds[best_f1_score_index]
            else:
                best_acc_score = np.max(accs[np.isfinite(accs)])
                best_acc_score_index = np.argmax(accs[np.isfinite(accs)])
                thresh = thresholds[best_acc_score_index]

            preds = test_preds
            thresh_preds = (test_preds > thresh).float()
            labels = test_labels
            acc = ((thresh_preds == labels).sum() / labels.shape[0]).cpu().item()
            rec = ((thresh_preds * labels).sum() / labels.sum()).cpu().item()
            pre = ((thresh_preds * labels).sum() / thresh_preds.sum()).cpu().item()
            auc = aucroc(preds, labels).item()
            f1 = rec * pre * 2 / (rec + pre) if rec + pre > 0 else 0

            auc_list.append(auc)
            f1_list.append(f1)
            acc_list.append(acc)

            s = 'class:' + str(cls) + ' val auc:' + str(round(val_auc, 4)) + ', test auc:' + str(round(auc, 4)) + ', val acc: ' \
                 + str(round(best_acc_score, 4)) + ', test f1: ' + str(round(f1, 4)) + ', test acc: ' + str(round(acc, 4))
            print(s)
            txt_rec.append(s)
            records['repeat_' + str(i)]['results_cls' + str(cls)] = {'val_auc': round(val_auc, 4), 'test_auc': round(auc, 4), \
                    'val_acc': round(best_acc_score, 4), 'test_f1': round(f1, 4), 'test_acc': round(acc, 4)}
            records['repeat_' + str(i)]['pred_cls' + str(cls)] = {'labels': labels.cpu().tolist(), \
                    'logits': preds.cpu().tolist(), 'preds': thresh_preds.cpu().tolist()}

    # ====================== count and record results ======================

    auc_mean = np.array(auc_list).mean()
    macro_auc = macro_value(auc_list, args.c)
    auc_std = np.array(macro_auc).std()
    f1_mean = np.array(f1_list).mean()
    macro_f1 = macro_value(f1_list, args.c)
    f1_std = np.array(macro_f1).std()
    acc_mean = np.array(acc_list).mean()
    macro_acc = macro_value(acc_list, args.c)
    acc_std = np.array(macro_acc).std()

    s = 'auc mean: ' + str(round(auc_mean, 4)) + ', auc std: ' + str(round(auc_std, 4)) + \
        ', f1 mean: ' + str(round(f1_mean, 4)) + ', f1 std: ' + str(round(f1_std, 4)) + \
        ', acc mean: ' + str(round(acc_mean, 4)) + ', acc std: ' + str(round(acc_std, 4))
    print(s)
    txt_rec.append(s)

    records['mean'] = {'auc_mean': round(auc_mean, 4), 'auc_std': round(auc_std, 4), 'auc_values': macro_auc, \
            'f1_mean': round(f1_mean, 4), 'f1_std': round(f1_std, 4), 'f1_values': macro_f1, \
            'acc_mean': round(acc_mean, 4), 'acc_std': round(acc_std, 4), 'acc_values': macro_acc}
    records['text_records'] = txt_rec

    return records


# ====================== the main function ======================

if __name__ == '__main__':

    # ====================== arg parser ======================

    parser = argparse.ArgumentParser('Multiple Instance Prompting')
    parser.add_argument('--mode', default='search', type=str, help="update: update features, inference: process query only, \
        eval: load processed features for evaluate, default: update and test")

    # hyper-params
    parser.add_argument('--topk', default=40, type=int, help='Number of top patchs to take')
    parser.add_argument('--top_instance', default=1, type=int, help='Number of top patchs to take')
    parser.add_argument('--temperature', default=10, type=float, help='Temperature for sample reweights')
    parser.add_argument('--related_thresh', default=0.88, type=float, help='cosine similarity threshold to select related patchs')
    parser.add_argument('--example_num', default=3, type=int, help='number of wsi for init example')
    parser.add_argument('--multiple_num', type=int, nargs='+', default=None, help='multi example num')

    # dataset information and settings
    parser.add_argument('--raw_feature_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--wsi_path', default='', type=str,
        help='Optional dir of raw WSI files. Required only for roughMask prompts, '
             '--seg, --vis_path, or --dump_pseudo. Pure classification on '
             'trident-extracted h5 features can leave this empty.')
    parser.add_argument('--dump_features', default=None, help='Path where to save features')
    parser.add_argument('--dump_pseudo', default='', help='Path where to save pseudo, vis and data split')
    parser.add_argument('--dump_records', default='', help='Path to save records (json file)')
    parser.add_argument('--vis_path', default='', help='Path where to save heatmap')
    parser.add_argument('--dataset_info', default='/path/to/data_list_gt_and_split', type=str, help='json file recording dataset info')
    parser.add_argument('--patch_scale', default=512, type=int, help='patch size in 40x for anno loading')
    parser.add_argument('--file_min_size', default=5000, type=int, help='skip background and patches with a few content')
    parser.add_argument('--c', default=1, type=int, help='number of class, c >1 for subtyping')
    parser.add_argument('--seg', default=False, action='store_true', help='True to evaluate segmentation task (f1 = dice)')

    # for weak prompts
    parser.add_argument('--prompt_type', default='mask', help='prompttation type')
    parser.add_argument('--prompt_path', default='', help='path of prompttation xml file')
    parser.add_argument('--ignore', default=0, type=float, help='degree to ignore uncertain example (during generating example)')
    parser.add_argument('--ignore_query', default=0.3, type=float, help='degree to ignore uncertain foreground query (subtyping only)')

    # test settings
    parser.add_argument('--seed', default=1024, type=int, help='for the reproduce of data split')
    parser.add_argument('--runs', default=5, type=int, help='number of test times')
    parser.add_argument('--val_num', default=100, type=int, help='number of validation WSIs')
    parser.add_argument('--test_num', default=129, type=int, help='number of test WSIs')
    parser.add_argument('--val_ratio', default=-1, type=float, help='split val test via ratio to replace specific number')
    parser.add_argument('--num_workers', default=4, type=int,
        help='DataLoader workers used to parallelise per-query slide pickle decode. '
             '0 disables multi-process loading. Support (example) slides are always '
             'cached in the main process and never re-decoded.')
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.dump_features, exist_ok=True)

    # collect features and information
    feature_processor(args)

    # ====================== Execute different modes ======================
    
    # evaluat with given hyper-parameters (in deployment)
    if args.mode == 'eval':
        print(args)
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            random.seed(args.seed)
            args.example_num = p
            res, rec = evaluate(args)
            records[str(p) + '-shot'] = rec

        if args.dump_records != '':
            np.save(args.dump_records, records)
    
    # run baselines
    if args.mode == 'baselines':
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            args.example_num = p
            records[str(p) + '-shot'] = {}

            # segmentation need patch predictions, knn is conducted on wsi-level
            if not args.seg and args.vis_path == '':
                print('mode: knn_mean, example ' + str(args.example_num))
                random.seed(args.seed)
                rec_knn_mean = evaluate_baseline(args, 'knn_mean')
                records[str(p) + '-shot']['knn_mean'] = rec_knn_mean
            
                print('mode: knn_max, example ' + str(args.example_num))
                random.seed(args.seed)
                rec_knn_max = evaluate_baseline(args, 'knn_max')
                records[str(p) + '-shot']['knn_max'] = rec_knn_max
            
            print('mode: prototype, example ' + str(args.example_num))
            random.seed(args.seed)
            rec_proto = evaluate_baseline(args, 'prototype')
            records[str(p) + '-shot']['prototype'] = rec_proto

            print('mode: prototype_simple_shot, example ' + str(args.example_num))
            random.seed(args.seed)
            rec_simp = evaluate_baseline(args, 'prototype_simple_shot')
            records[str(p) + '-shot']['simple_Shot'] = rec_simp

        if args.dump_records != '':
            np.save(args.dump_records, records)

    # run in val-test set with hyperparameter search
    if args.mode == 'default': 

        pseudo = args.dump_pseudo
        args.dump_pseudo = ''

        # speed up param search
        if args.c > 1:
            ori_runs = args.runs
            ori_val_num = args.val_num
            args.runs=3
            args.val_num=50
        
        # search for parameters (in extended data figure 10)
        if args.c > 1:
            v, t = 0, 0
            for p in [1000, 2000, 3000, 4000, 5000]:
                print('searching top_instance, param: ' + str(p))
                random.seed(args.seed) # validate params without influence from sampling
                args.top_instance = p
                res, _ = evaluate(args, val_only=True)
                if res > v:
                    v = res
                    t = p
            args.top_instance = t
            print('params: top_instance, searched threshold: ' + str(t) + ', mean:' + str(v))

        v, t = 0, 0
        for p in [0, 0.02, 0.04, 0.06, 0.08]:
            print('searching ignore, param: ' + str(p))
            random.seed(args.seed)
            args.ignore = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.ignore = t
        print('params: ignore, searched threshold: ' + str(t) + ', mean:' + str(v))

        if args.c > 1:
            v, t = 0, 0
            for p in [0.1, 0.15, 0.2, 0.25, 0.3]:
                print('searching ignore-query, param: ' + str(p))
                random.seed(args.seed)
                args.ignore_query = p
                res, _ = evaluate(args, val_only=True)
                if res > v:
                    v = res
                    t = p
            args.ignore_query = t
            print('params: ignore-query, searched threshold: ' + str(t) + ', mean:' + str(v))

        v, t = 0, 0
        for p in [20, 30, 40, 50, 60]:
            print('searching topk, param: ' + str(p))
            random.seed(args.seed)
            args.topk = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.topk = t
        print('params: topk, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [0.86, 0.87, 0.88, 0.89, 0.9]:
            print('searching related_thresh, param: ' + str(p))
            random.seed(args.seed)
            args.related_thresh = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.related_thresh = t
        print('params: related_thresh, searched threshold: ' + str(t) + ', mean:' + str(v))
            
        v, t = 0, 0
        for p in [1, 5, 10, 20, 30]:
            print('searching temperature, param: ' + str(p))
            random.seed(args.seed)
            args.temperature = p
            res, _ = evaluate(args, val_only=True)
            if res > v:
                v = res
                t = p
        args.temperature = t
        print('params: temperature, searched threshold: ' + str(t) + ', mean:' + str(v))
        
        # eval with searched params and test influence of example number
        if args.c > 1:
            args.runs = ori_runs
            args.val_num = ori_val_num
        args.dump_pseudo = pseudo

        print(args)
        records = {}
        num = [args.example_num] if args.multiple_num == None else args.multiple_num
        for p in num:
            print('eval %d-shot:' % (p))
            random.seed(args.seed)
            args.example_num = p
            res, rec = evaluate(args)
            records[str(p) + '-shot'] = rec
        
        # save results
        if args.dump_records != '':
            np.save(args.dump_records, records)
