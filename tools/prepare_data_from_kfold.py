import pandas as pd
import os
import pathlib
import shutil

data_dir = '/home/nguyenhonghai/datasets_test/ABAW3/'


def copy_predict_fold():
    cur_path = os.getcwd()
    train_logs_path = os.path.join(os.path.dirname(cur_path), 'train_logs/VA_v1')
    dst = os.path.join(data_dir, 'extract_data_kfold')
    if not os.path.isdir(dst):
        os.mkdir(dst)

    # 5 folds
    for k in range(5):
        train_fold_path = pathlib.Path(os.path.join(train_logs_path, 'VA_fold{}/test'.format(k)))
        val_fold_path = pathlib.Path(os.path.join(train_logs_path, 'VA_fold{}/test/_1'.format(k)))
        val_test_path = pathlib.Path(os.path.join(train_logs_path, 'VA_fold{}/test/_1/_1'.format(k)))

        dst_fold = os.path.join(dst, 'kfold{}'.format(k))
        if not os.path.isdir(dst_fold):
            os.mkdir(dst_fold)

        for f in train_fold_path.glob('*csv'):
            shutil.copyfile(f, os.path.join(dst_fold, os.path.basename(f)))

        for f in val_fold_path.glob('*csv'):
            shutil.copyfile(f, os.path.join(dst_fold, os.path.basename(f)))

        for f in val_test_path.glob('*csv'):
            shutil.copyfile(f, os.path.join(dst_fold, os.path.basename(f)))


def extract_data():
    fold_path = pathlib.Path('{}extract_data_kfold/'.format(data_dir))
    features_path = pathlib.Path('{}extract_features/'.format(data_dir))
    if not os.path.isdir(features_path):
        os.mkdir(features_path)
    k0_path = pathlib.Path(fold_path, 'kfold0')

    for f in k0_path.glob('*csv'):
        print(f)

        basename = os.path.basename(f).replace('.csv', '')
        if not os.path.isdir(os.path.join(features_path, basename)):
            os.mkdir(os.path.join(features_path, basename))

        f1 = pathlib.Path(str(f).replace('kfold0', 'kfold1'))
        f2 = pathlib.Path(str(f).replace('kfold0', 'kfold2'))
        f3 = pathlib.Path(str(f).replace('kfold0', 'kfold3'))
        f4 = pathlib.Path(str(f).replace('kfold0', 'kfold4'))

        v = pd.read_csv(f).sort_values(by=['Index']).values
        v1 = pd.read_csv(f1).sort_values(by=['Index']).values
        v2 = pd.read_csv(f2).sort_values(by=['Index']).values
        v3 = pd.read_csv(f3).sort_values(by=['Index']).values
        v4 = pd.read_csv(f4).sort_values(by=['Index']).values

        v_v = v[:, 1]
        v_a = v[:, 2]

        v1_v = v1[:, 1]
        v1_a = v1[:, 2]

        v2_v = v2[:, 1]
        v2_a = v2[:, 2]

        v3_v = v3[:, 1]
        v3_a = v3[:, 2]

        v4_v = v4[:, 1]
        v4_a = v4[:, 2]

        w = 1
        w1 = 1
        w2 = 1
        w3 = 1
        w4 = 1

        dct = {
            'index': v[:, 0].astype(int),
            'v_v': v_v * w, 'v1_v': v1_v * w1, 'v2_v': v2_v * w2, 'v3_v': v3_v * w3, 'v4_v': v4_v * w4,
            'v_a': v_a * w, 'v1_a': v1_a * w1, 'v2_a': v2_a * w2, 'v3_a': v3_a * w3, 'v4_a': v4_a * w4,
            'V': v[:, 3], 'A': v[:, 4]
        }  # 0.4856-w=1

        df = pd.DataFrame(data=dct)
        df.to_csv(os.path.join(features_path, basename, 'combine_features_va.csv'), index=False)
        #

copy_predict_fold()
extract_data()