import os.path as osp
import numpy as np
from PIL import Image
from pathlib import Path
import mmcv

# convert dataset annotation to semantic segmentation map
data_root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\dataprocess'
# img_dir = 'images'
ann_dir = 'resized_labels'
color_dir = 'colors'
# define class and plaette for better visualization
classes = ('road', 'building', 'tree', 'car', 'traffic', 'other')

palette = [[255, 0, 209], [255, 204, 0], [6, 255, 0], [0, 0, 255],
           [219, 24, 22], [43, 37, 67]]

reduced_folder_path = Path(data_root) / ann_dir

for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.png'):
    image = Image.open(reduced_folder_path / file)
    image = image.convert("L")
    # image = image.resize(size=(512, 256), resample=Image.NEAREST)
    seg_map = np.array(image)
    # seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))

    # change filename to match the name of raw image
    temp = file.split('_')
    temp[3] = 'leftImg8bit.png'
    temp.pop()
    temp = '_'.join(temp)

    seg_img.save(osp.join(data_root, color_dir, temp))
