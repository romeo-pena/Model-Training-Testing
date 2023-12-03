import mmcv
import os.path as osp

data_root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\fc_dataprocess'
ann_dir = 'color'

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(data_root, ann_dir), suffix='.png')]

# select first 4/5 as train set
train_length = int(len(filename_list) * 4 / 5)

with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    f.writelines(line + '\n' for line in filename_list[:train_length])
with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    f.writelines(line + '\n' for line in filename_list[train_length:])
