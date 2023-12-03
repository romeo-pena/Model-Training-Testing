import numpy as np
from PIL import Image
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path

data_root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\dataprocess'
color_dir = 'colors'
# define class and plaette for better visualization
classes = ('road', 'building', 'tree', 'car', 'traffic', 'other')

palette = [[255, 0, 209], [255, 204, 0], [6, 255, 0], [0, 0, 255],
           [219, 24, 22], [43, 37, 67]]

color_folder_path = Path(data_root) / color_dir
label_names = sorted(color_folder_path.glob("*"))
test_index = 5
image_path = label_names[test_index]

img = Image.open(image_path)
plt.figure(figsize=(14, 6))
im = plt.imshow(np.array(img.convert('RGB')))

# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=np.array(palette[i])/255.,
                          label=classes[i]) for i in range(6)]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           fontsize='large')

plt.show()
