from pathlib import Path
from PIL import Image
import numpy as np
from helper import reduce_id_labels, reduce_id_labels_sky

label_folder = 'labelids'
save_folder = 'reduced_labels_sky'
root = r'C:\Users\ERDT\PycharmProjects\MMSeg\mmsegmentation\dataprocess'

label_folder_path = Path(root) / label_folder
reduced_folder_path = Path(root) / save_folder

label_names = sorted(label_folder_path.glob("*"))

for item in label_names:
    image_path = item

    with open(image_path, "rb") as image_file:
        image = Image.open(image_file)
        image = image.convert("L")

    image_name = str(image_path).split('\\')
    image_name = image_name[-1]

    image = np.array(image)
    image = reduce_id_labels_sky(image)

    im = Image.fromarray(image)
    save_name = reduced_folder_path / image_name
    im.save(save_name)

# print(np.unique(image))
