import numpy as np
import os
import jsonlines as json


def parse_line(sample):
    """Parse an ndjson line and return ink (as np array) and classname."""
    class_name = sample["word"]
    inkarray = sample["drawing"]
    stroke_lengths = [len(stroke[0]) for stroke in inkarray]
    total_points = sum(stroke_lengths)
    np_ink = np.zeros((total_points, 3), dtype=np.float32)
    current_t = 0
    for stroke in inkarray:
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
    np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]
    return np_ink, class_name


DEFAULT_SEED = 22012018
rng = np.random.RandomState(DEFAULT_SEED)
num_examples = 30000

train_ratio = 0.7
valid_ratio = 0.15

dir_path = "quickdraw_data/strokes"

file_list = os.listdir(dir_path)

inputs = list()
targets = list()
for file in file_list:
    file_path = os.path.join(
        dir_path, file)

    i = 0
    with json.open(file_path) as reader:
        for obj in reader:
            if i >= num_examples:
                break
            i += 1
            inp, tar = parse_line(obj)
            inputs.append(inp)
            targets.append(tar)

inputs = np.array(inputs)
targets = np.array(targets)

idx = np.arange(len(inputs))
rng.shuffle(idx)

inputs = inputs[idx]
targets = targets[idx]

train_size = int(len(inputs)*train_ratio)
valid_size = int(len(inputs)*valid_ratio)

train_features = inputs[0:train_size]
train_classes = targets[0:train_size]
valid_features = inputs[train_size:(train_size+valid_size)]
valid_classes = targets[train_size:(train_size+valid_size)]
test_features = inputs[(train_size+valid_size):]
test_classes = targets[(train_size+valid_size):]

np.savez('data/quickdraw-train', inputs=train_features, targets=train_classes)
np.savez('data/quickdraw-valid', inputs=valid_features, targets=valid_classes)
np.savez('data/quickdraw-test', inputs=test_features, targets=test_classes)
