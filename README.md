# Model-Training-Testing
This is the repository used to train and run the DeepLabV3+ models. This project uses an old version of MMSegmentation to train and test a DeepLabV3+ model for our research. This project was used in pyCharm and we recommend using pyCharm as the IDE of choice for using this project

## Software Requirements
- Python 3.8
- Pytorch
- CUDA Drivers

## Hardware Requirements
- GPU with 12gb of vram

## Setup
1. Install requirements by running commands `pip install -r requirements.txt` and `pip install -r requirements_all.txt`
2. Run python `python setup.py'` to install the old version of MMSegmentation
3. Once done you can start running the project

## How to use
1. Becuase we use the MMSegmentation library, we utilize configuration to train our models. Our inital model configuration can be found in `cfgs/deeplabv3plus_base`
2. Once ready, we intialize our training by running the `train.py` script.
    - This script utilizes the following parameters to run:
       - --work_dir : The directory the runner saves the checkpoints and logs
       - --data_dir : The directory of the dataset to be used for training and testing
       - --config_file : the CFG file, usually the one found in our folder
       - --checkpoint : the checkpoint file if you want to continue running from a specific checkpoint
       - An example run script would look like this `python train.py --work_dir work_dirs/FCG_Base --data_dir datasets/FCG_Base --config_file cfgs/deeplabv3plus_base.py`
    - While running this script, we recommend running the `log_reader.py` command as it reads the logs made in the work_dir and shows recommendations on when to stop running the training sequence
        - An example run script would look like this `python log_reader.py --log_file work_dirs/FCG_Base/None.log.json --threshold 6`
        - The parameters are only the log file and the threshold of how many times we can go below our highest metrics before recommending the stop for training
3. After Running `train.py`, we run our `eval.py` script for evaluation and inferencing.
     - Like the previous script this script utilizes similar parameters as the parameters are:
       - --work_dir : The directory the runner saves the checkpoints and logs
       - --data_dir : The directory of the dataset to be used for training and testing
       - --config_file : the CFG file, usually the one found in our folder
       - --checkpoint : the checkpoint file if you want to continue running from a specific checkpoint
       - --inference : a boolean value to make inference image on the working directory
       - --iterations : the iteration value at which the checkpoint is
       - An example run script would look like this `python eval.py --work_dir exploratory/FCG_Base  --data_dir datasets/FCG_Base --config_file cfgs/deeplabv3plus_base.py --checkpoint exploratory/FCG_Base/iter_40000.pth --iterations 40000 --inference`
      
## Dataset Structure
1. The dataset structure is the structure of the folder we expect the data to be in. This is what we pass to the --data_dir parameter in the train and eval scripts.
- Dataset_root (name of dataset)
    - images (directory with rgb images)
    - labels (directory with ground truth images)
    - splits (directory containining txt files containing lists of the image names for testing and training)
        - train.txt
        - test.txt
2. Additionally, we add the cityscapes images as well into the dataset when we run the `eval.py` script. we add the following folders into the dataset_root directory
    - cimages (directory with cityscapes rgb images)
    - clabels (directory with ground truth images)
    - csplits (directory containining txt files containing lists of the image names for testing and training)
 
## Processing the dataset
1. There are 2 sub directories in this project that were used to help process and create the datasets used in this project. The two folders are as follows:
    1. the `dataprocess` folder - the directory to process the cityscapes dataset into 6 classes
    2. the `fc_dataprocess` folder - the directory to process the dataset created by the PCG tool
2. Info about running their processing is found in their respective folders. 
