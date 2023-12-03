import os.path as osp
import numpy as np
from PIL import Image
from pathlib import Path
import mmcv


def rgb2mask(img):
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3

    W = np.power(256, [[0], [1], [2]])

    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)

    mask = np.zeros(img_id.shape)

    for i, c in enumerate(values):
        try:
            mask[img_id == c] = palette_index[tuple(img[img_id == c][0])]
        except:
            pass
    return mask


# convert dataset annotation to semantic segmentation map
data_root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\fc_dataprocess'
color_dir = 'color'
save_dir = 'color_new'
# # define class and plaette for better visualization
# classes = ('car', 'road', 'building', 'tree', 'other', 'traffic')

# palette = [[0, 0, 255], [255, 0, 209], [255, 204, 0], [6, 255, 0],
#            [43, 37, 67], [219, 24, 22]]

palette = [[255, 0, 209], [255, 204, 0], [6, 255, 0], [0, 0, 255],
            [219, 24, 22], [43, 37, 67]]

palette_index = {(255, 0, 209): 0,
                 (255, 204, 0): 1,
                 (6, 255, 0): 2,
                 (0, 0, 255): 3,
                 (219, 24, 22): 4,
                 (43, 37, 67): 5,
                 (0, 0, 0): 5}

color_folder_path = Path(data_root) / color_dir
root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\fc_dataprocess'
mmcv.mkdir_or_exist(osp.join(root, save_dir))

for file in mmcv.scandir(osp.join(data_root, color_dir), suffix='.png'):
    image = Image.open(color_folder_path / file)
    image = image.convert("RGB")
    # image = rgb2mask(image)

    seg_map = np.array(image)
    seg_map = rgb2mask(seg_map)
    # seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))

    seg_img.save(osp.join(data_root, save_dir, file))
