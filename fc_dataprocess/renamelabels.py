import mmcv
import os.path as osp
from PIL import Image
from pathlib import Path

root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\fc_dataprocess'
color_raw_dir = 'color_raw'
rgb_raw_dir = 'rgb_raw'
color_dir = 'color'
rgb_dir = 'rgb'

color_raw_dir_path = Path(root) / color_raw_dir
mmcv.mkdir_or_exist(osp.join(root, color_dir))
# rename labels in color folder
for file in mmcv.scandir(osp.join(root, color_raw_dir), suffix='.png'):

    temp = file.split('_')
    temp[0] = 'data'
    temp = '_'.join(temp)

    image = Image.open(color_raw_dir_path / file)
    image.save(osp.join(root, color_dir, temp))

rgb_raw_dir_path = Path(root) / rgb_raw_dir
mmcv.mkdir_or_exist(osp.join(root, rgb_dir))
# rename labels in rgb folder
for file in mmcv.scandir(osp.join(root, rgb_raw_dir), suffix='.png'):

    temp = file.split('_')
    temp[0] = 'data'
    temp = '_'.join(temp)

    image = Image.open(rgb_raw_dir_path / file)
    image.save(osp.join(root, rgb_dir, temp))

