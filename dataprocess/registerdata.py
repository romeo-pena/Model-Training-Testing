from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

classes = ('road', 'building', 'tree', 'car', 'traffic', 'other')

palette = [[255, 0, 209], [255, 204, 0], [6, 255, 0], [0, 0, 255],
           [219, 24, 22], [43, 37, 67]]


@DATASETS.register_module()
class CityScapesCustomDataset(CustomDataset):
    def format_results(self, results, imgfile_prefix, indices=None, **kwargs):
        pass

    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
