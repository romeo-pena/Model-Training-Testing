## How to use
1. We assume we have the folders `rgb_raw` and `color_raw` containing the raw rgb images and label images from the PCG tool. 
2. We then run `renamelabels.py` to rename the files found in the `rgb_raw` and `color_raw` folders for easier file reading
3. We then run `processlabels.py` to fix the labeling of the labeled imgaes
4. We then use `datasplits.py` to create a splits folder to create a `train.txt` and `val.txt` files
5. Once finished, transfer the new folders (`rgb`, `color`, `splits`) into another directory into this project for dataset orgranization.
6. Don't forget to rename the `rgb` and `color` directories to `images` and `labels` if you want to use the datasets for training and evaluation
