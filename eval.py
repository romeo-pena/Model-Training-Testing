import mmcv
import warnings
import argparse
import os.path as osp
import copy

from mmseg.datasets import build_dataset
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, inference_segmentor

# from configs import getConfigFCG, getConfig
from cfg_loader import getConfig_base, getConfig_base_train
from archive.custom_util.utils import save_result


# REGISTER THE DATASET
def parseargs():
    parser = argparse.ArgumentParser(description='Evaluate a segmentor')
    parser.add_argument('--work_dir', help='The working directory')
    parser.add_argument('--data_dir', help='The directory containing the dataset')
    parser.add_argument('--config_file', help='Config file to be used')
    parser.add_argument('--checkpoint', help='Checkpoint file of config')
    parser.add_argument('--inference', action='store_true', help='Whether to do an inference')
    parser.add_argument('--iterations', help='iteration value for evaluation of the checkpoint')

    args = parser.parse_args()
    return args


def main():
    warnings.filterwarnings("ignore")
    args = parseargs()

    # Variables
    work_dir = args.work_dir
    data_dir = args.data_dir
    config_file = args.config_file
    load_pth = args.checkpoint
    inf_flag = args.inference
    iterations = int(args.iterations)

    assert isinstance(work_dir, str)
    assert isinstance(data_dir, str)
    assert isinstance(config_file, str)
    assert isinstance(load_pth, str)
    assert isinstance(iterations, int)

    inference_dir = './work_dirs/' + work_dir + '/inference/'

    # Get Validation Images
    val_list = []
    with open(data_dir + '/csplits/val.txt') as f:
        for line in f:
            val_list.append(line.strip())

    # val_path = "work_dirs/" + work_dir + "/val"
    print("Iteration value at: " + str(iterations))
    # Set up the model with config
    cfg = getConfig_base_train(config_file, data_dir, load_pth, work_dir, mode='Eval', iterations=iterations)
    # cfg = None

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    model = build_segmentor(cfg.model)
    model.CLASSES = datasets[0].CLASSES
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # Eval and Inference (Optional)
    train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                    meta=dict())

    # Do an inference if inference flag is true
    palette = [[255, 0, 209], [255, 204, 0], [6, 255, 0], [0, 0, 255],
               [219, 24, 22], [43, 37, 67]]



    if inf_flag:
        model.eval()
        model.cfg = cfg
        for image in val_list:
            img_path = data_dir + '/cimages/' + str(image) + ".png"
            img = mmcv.imread(img_path)
            result = inference_segmentor(model, img)

            out_file = work_dir + '/inference/' + str(image) + ".png"
            save_result(model, img, result, palette, out_file=out_file)
            # show_result_pyplot(model, img, result, palette)


@DATASETS.register_module()
class CustomExperimentDataset(CustomDataset):
    CLASSES = ('road', 'building', 'tree', 'car', 'traffic', 'other')
    PALETTE = [[255, 0, 209], [255, 204, 0], [6, 255, 0], [0, 0, 255],
               [219, 24, 22], [43, 37, 67]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


if __name__ == '__main__':
    main()
