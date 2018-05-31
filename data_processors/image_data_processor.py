import numpy as np
import os

"""
This code is used for initial processing of the full image data.
It takes the first 25k examples from each file and stores them.
The labels are in order in the respective file.
"""


file_names = sorted(os.listdir(os.path.join('quickdraw_data', 'images')))

processed_data = []
inputs = np.empty((0, 784), dtype=np.uint8)
class_names = []

for file_name in file_names:
    data_path = os.path.join('quickdraw_data', 'images', file_name)
    # load data from compressed numpy file
    assert os.path.isfile(data_path), (
    'Data file does not exist at expected path: ' + data_path
    )
    class_name = file_name.split('%2F')[-1].split('.')[0]
    loaded = np.load(data_path)
    assert len(loaded) > 24999, (
    'File {} does not have enough data.'.format(file_name)
    )
    num_examples = 25000
    inputs = np.append(inputs, loaded[:num_examples], axis=0)
    class_names.append(class_name)

inputs = np.reshape(inputs, newshape=(-1, 28, 28))

np.save('quickdraw_data/image_data_100', inputs)
np.save('quickdraw_data/image_data_100_classes', class_names)
