import numpy as np
import os
import jsonlines as json

"""
This code is used for initial processing of the full strokes data.
It takes the first 25k examples from each file and stores them.
The labels are in order in the respective file.
"""

def parse_line(sample):
    class_name = sample["word"]
    if not class_name:
        print ("Empty classname")
        return None, None
    inkarray = sample["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    if not inkarray:
        print("Empty inkarray")
        return None, None
    for stroke in inkarray:
        if len(stroke[0]) != len(stroke[1]):
            print("Inconsistent number of x and y coordinates.")
            return None, None
        for i in [0, 1]:
            np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
        current_t += len(stroke[0])
        np_ink[current_t - 1, 2] = 1  # stroke_end
    # Preprocessing.
    # 1. Size normalization.
    lower = np.min(np_ink[:, 0:2], axis=0)
    upper = np.max(np_ink[:, 0:2], axis=0)
    scale = upper - lower
    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]
    return np_ink, class_name


file_names = sorted(os.listdir(os.path.join('..', 'quickdraw_data', 'strokes')))
class_names = []
inputs = list()

for file_name in file_names:
    data_path = os.path.join('..', 'quickdraw_data', 'strokes', file_name)
    # load data from compressed numpy file
    assert os.path.isfile(data_path), (
    'Data file does not exist at expected path: ' + data_path
    )

    i = 0
    num_examples = 25000
    with json.open(data_path) as reader:
        for obj in reader:
            if i >= num_examples:
                break
            i += 1
            ink, tar = parse_line(obj)
            if ink is None:
                print ("Couldn't parse ink from '" + str(i) + "'.")
            if ink.shape[0] > 70:
                result = ink[:70, : ]
            else:
                result = np.zeros((70, 3))
                result[:ink.shape[0], :] = ink
            inputs.append(result)
    class_name = file_name.split('%2F')[-1].split('.')[0]
    class_names.append(class_name)

inputs = np.array(inputs, dtype=np.float16)
np.save('../quickdraw_data/strokes_data_100', inputs)
np.save('../quickdraw_data/strokes_data_100_classes', class_names)
