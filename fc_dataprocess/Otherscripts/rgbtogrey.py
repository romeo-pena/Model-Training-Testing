import mmcv
import os.path as osp
from PIL import Image
from pathlib import Path

root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\fc_dataprocess'
grey_dir = 'grey'
rgb_dir = 'rgb'

rgb_dir_path = Path(root) / rgb_dir
mmcv.mkdir_or_exist(osp.join(root, grey_dir))
# rename labels in color folder
for file in mmcv.scandir(osp.join(root, rgb_dir), suffix='.png'):
    image = Image.open(rgb_dir_path / file)
    image = image.convert('L')
    image.save(osp.join(root, grey_dir, file))


