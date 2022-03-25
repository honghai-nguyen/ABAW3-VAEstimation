import os.path
import sys

import pandas as pd
import numpy as np
import pathlib
import argparse
from tqdm import tqdm
import shutil

from sklearn.model_selection import KFold


def parse_video_annos(anno_inp, video_ids, discard_value, drop_all=False):
    if isinstance(anno_inp, str):
        anno_data = pd.read_csv(anno_inp, skiprows=1, header=None)
    elif not isinstance(anno_inp, pd.DataFrame):
        raise ValueError('Anno data must be a DataFrame or path to anno file.')
    else:
        anno_data = anno_inp

    anno_values = anno_data.values
    if isinstance(discard_value, int):
        use_indexes = np.argwhere(np.sum(anno_values == discard_value, axis=1) == 0).flatten()
        anno_indexes = use_indexes

    elif isinstance(discard_value, str) and discard_value == 'mtl':
        va_indexes = np.sum(anno_values[:, 1:3] == -5, axis=1) == 0
        expr_indexes = np.sum(anno_values[:, 3].reshape(-1, 1) == -1, axis=1) == 0
        au_indexes = np.sum(anno_values[:, 4:16] == -1, axis=1) == 0

        current_indexes = np.array([int(x.split('/')[-1][:-4]) for x in anno_values[:, 0]]) - 1
        # Keep index if at least 1 of 3 task is annotated
        anno_indexes = np.argwhere(np.logical_or(np.logical_or(va_indexes, expr_indexes), au_indexes)).flatten()

        use_indexes = current_indexes[anno_indexes].flatten()
        # Remove duplicate values
        val_use, val_use_index = np.unique(use_indexes, return_index=True)
        anno_indexes = anno_indexes[val_use_index]
        use_indexes = use_indexes[val_use_index]
    else:
        raise ValueError('Unknown discard value of {}'.format(discard_value))

    # print(anno_values.shape, video_ids)
    cropped_aligned_images = sorted((cropped_aligned / video_ids).glob('*.jpg'))
    frames_idxes = np.array([int(x.name[:-4]) - 1 for x in cropped_aligned_images])

    if drop_all:
        # Drop frames that do not have labels, and labels that do not have corresponding frames
        keep_indexes = sorted(set(frames_idxes).intersection(set(use_indexes)))
        if discard_value != 'mtl':
            anno_values = anno_values[keep_indexes, :]
            keep_frames = np.array(['{}/{}.jpg'.format(video_ids, str(x + 1).zfill(5)) for x in keep_indexes]).reshape(
                -1, 1)
            ret_data = np.hstack([keep_frames, anno_values, np.array(keep_indexes).reshape(-1, 1)])
        else:

            if len(use_indexes) > len(keep_indexes):
                for idx in range(len(use_indexes)):
                    if use_indexes[idx] not in keep_indexes:
                        anno_indexes[idx] = -1

                anno_indexes = anno_indexes[anno_indexes > -1]

            ret_data = np.hstack([anno_values[anno_indexes, :], np.array(keep_indexes).reshape(-1, 1)])


    else:
        keep_frames = []
        for idx in use_indexes:
            if idx not in frames_idxes:
                # TODO: Mixup instead of copy
                copy_index = frames_idxes[np.argmin(np.abs(frames_idxes - idx))]
            else:
                copy_index = idx
            keep_frames.append('{}/{}.jpg'.format(video_ids, str(copy_index + 1).zfill(5)))

        if discard_value != 'mtl':
            anno_values = anno_values[anno_indexes, :]
        else:
            anno_values = anno_values[anno_indexes, 1:]
        keep_frames = np.array(keep_frames).reshape(-1, 1)
        ret_data = np.hstack([keep_frames, anno_values, use_indexes.reshape(-1, 1)])

    return ret_data


def split_annotation(challenges, n_folds, root_dir):
    kfold_foler = root_dir / 'Kfold_Annotations'
    if not os.path.isdir(kfold_foler):
        os.mkdir(kfold_foler)
    else:
        shutil.rmtree(kfold_foler)
    for chal in challenges:
        chal_name = chal.name.split('_')[0]
        chal_dataset = {}
        skf = KFold(n_splits=n_folds, random_state=5, shuffle=True)
        if 'MTL' not in chal_name:
            for data_type in ['Train']:
                data_list = sorted(list((chal / '{}_Set'.format(data_type)).glob('*.txt')))
                chal_dataset[data_type] = dict()
                count = 0
                for tr_ind, vl_ind in skf.split(X=data_list):
                    sub_kfold = kfold_foler / 'kfold{}'.format(str(count))
                    sub_kfold_train = sub_kfold / chal.name / 'Train_Set'
                    sub_kfold_val = sub_kfold / chal.name / 'Validation_Set'

                    if not os.path.exists(sub_kfold_train):
                        os.makedirs(sub_kfold_train)
                    else:
                        for f in sub_kfold_train.glob('*'):
                            os.unlink(f)

                    if not os.path.exists(sub_kfold_val):
                        os.makedirs(sub_kfold_val)
                    else:
                        for f in sub_kfold_val.glob('*'):
                            os.unlink(f)

                    for f in np.array(data_list)[tr_ind]:
                        des_file = sub_kfold_train / os.path.basename(f)
                        shutil.copyfile(f, des_file)
                    for f in np.array(data_list)[vl_ind]:
                        des_file = sub_kfold_val / os.path.basename(f)
                        shutil.copyfile(f, des_file)
                    count += 1

        else:
            print("Do not deploy MTL!!!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data preparation ABAW3 - CVPR 2022')
    parser.add_argument('--root_dir', type=str, default='/home/nguyenhonghai/datasets_test/ABAW3/', help='Root data folder')
    parser.add_argument('--out_dir', type=str, default='/home/nguyenhonghai/datasets_test/ABAW3/', help='Out data folder')
    args = parser.parse_args()

    root_dir = pathlib.Path(args.root_dir)
    cropped_aligned = root_dir / 'cropped_aligned'
    annotations = root_dir / '3rd_ABAW_Annotations'

    discard_values = {'AU': -1, 'VA': -5, 'EXPR': -1}

    challenges = sorted([x for x in annotations.iterdir() if x.is_dir()])
    n_folds = 5
    drop_all_invalid = True
    # k-fold
    split_annotation(challenges, n_folds, root_dir)

    for chal in challenges:
        chal_name = chal.name.split('_')[0]
        chal_dataset = {}
        for k in range(n_folds):
            print(f"Kfold{k}")
            sub_kfold_anno = root_dir / 'Kfold_Annotations' / 'kfold{}'.format(str(k))
            sub_kfold_data = root_dir / 'Kfold_Data' / 'kfold{}'.format(str(k))
            if not os.path.exists(sub_kfold_data):
                os.makedirs(sub_kfold_data)

            if 'MTL' not in chal_name:
                for data_type in ['Train', 'Validation']:
                    data_list = sorted(list((sub_kfold_anno / chal.name / '{}_Set'.format(data_type)).glob('*.txt')))

                    chal_dataset[data_type] = dict()
                    for vd in tqdm(data_list):
                        parsed_data = parse_video_annos(vd.__str__(), vd.name[:-4], discard_value=discard_values[chal_name],
                                                        drop_all=drop_all_invalid)
                        chal_dataset[data_type][vd.name[:-4]] = parsed_data
            else:
                for data_type in ['Train', 'Validation']:
                    chal_dataset[data_type] = dict()
                    data_list = pd.read_csv(chal / '{}_set.txt'.format(data_type.lower()), skiprows=1, header=None)
                    data_list[16] = data_list.apply(lambda row: row[0].split('/')[0], axis=1)

                    for group_key, group_df in tqdm(data_list.groupby(by=16)):
                        parsed_data = parse_video_annos(group_df, group_key, discard_value='mtl', drop_all=drop_all_invalid)
                        chal_dataset[data_type][group_key] = parsed_data

            np.save((sub_kfold_data / '{}.npy'.format(chal_name)).__str__(), chal_dataset)

    for chal in challenges:
        chal_name = chal.name.split('_')[0]
        print(chal_name)
        chal_dataset = {}
        if 'MTL' not in chal_name:
            for data_type in ['Train', 'Validation']:
                data_list = sorted(list((chal / '{}_Set'.format(data_type)).glob('*.txt')))
                chal_dataset[data_type] = dict()
                for vd in tqdm(data_list):
                    parsed_data = parse_video_annos(vd.__str__(), vd.name[:-4], discard_value=discard_values[chal_name],
                                                    drop_all=drop_all_invalid)
                    chal_dataset[data_type][vd.name[:-4]] = parsed_data
        else:
            for data_type in ['Train', 'Validation']:
                chal_dataset[data_type] = dict()
                data_list = pd.read_csv(chal / '{}_set.txt'.format(data_type.lower()), skiprows=1, header=None)
                data_list[16] = data_list.apply(lambda row: row[0].split('/')[0], axis=1)

                for group_key, group_df in tqdm(data_list.groupby(by=16)):
                    parsed_data = parse_video_annos(group_df, group_key, discard_value='mtl', drop_all=drop_all_invalid)
                    chal_dataset[data_type][group_key] = parsed_data

        np.save((pathlib.Path(args.out_dir) / '{}.npy'.format(chal_name)).__str__(), chal_dataset)
