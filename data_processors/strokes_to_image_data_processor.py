import numpy as np
import os
import jsonlines as json
from skimage.draw import line_aa
import matplotlib.pyplot as plt
from scipy.misc import imresize


"""
This code is used for initial processing of the full strokes data.
It takes the first 25k examples from each file and stores them.
The labels are in order in the respective file.
"""

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


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

    lower = np.min(np_ink[:, 0:2], axis=0).astype(np.uint32)
    upper = np.max(np_ink[:, 0:2], axis=0).astype(np.uint32)
    scale = upper - lower

    max_axis = np.argmax(scale)
    stroke_ints = np_ink.astype(np.uint8)
    #import pdb; pdb.set_trace()
    offset = (scale[max_axis] - scale[1-max_axis]) // 2
    stroke_ints[:, 0:2] = stroke_ints[:, 0:2] - lower
    stroke_ints[:, 1-max_axis] = stroke_ints[:, 1-max_axis] + offset
    image = np.zeros((scale[max_axis]+1, scale[max_axis]+1), dtype=np.uint8)

    for point in range(1, len(stroke_ints)):
        if stroke_ints[point-1, 2] == 1:
            continue
        rr, cc, val = line_aa(stroke_ints[point-1, 1], stroke_ints[point-1, 0], stroke_ints[point, 1], stroke_ints[point, 0])
        if max(rr) >= scale[max_axis]+1 or max(cc) >= scale[max_axis]+1:
            import pdb; pdb.set_trace()
        image[rr, cc] = val * 255

    image = imresize(image, (28, 28))

    scale[scale == 0] = 1
    np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
    # 2. Compute deltas.
    np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
    np_ink = np_ink[1:, :]

    return np_ink, image, class_name


#dir_path = '/Users/choobea/mlpcw/mlp_cw_3_4/quickdraw_data'
#file_names = sorted(os.listdir(dir_path))
#file_names = sorted(os.listdir(os.path.join('..', 'quickdraw_data', 'strokes')))

dir_path = '/Users/choobea/mlpcw/mlp_cw_3_4/quickdraw_data'
file_names = sorted(os.listdir(os.path.join(dir_path, 'strokes')))

class_names = []
images = list()
inputs = list()

j = 0
for file_name in file_names:
    if j >= 1:
        break
    j +=1
    data_path = os.path.join(dir_path, 'strokes', file_name)
    # load data from compressed numpy file
    assert os.path.isfile(data_path), (
            'Data file does not exist at expected path: ' + data_path
    )

    i = 0
    num_examples = 10
    with json.open(data_path) as reader:
        for obj in reader:
            if i >= num_examples:
                break
            i += 1
            ink, img, tar = parse_line(obj)

            if ink is None:
                print ("Couldn't parse ink from '" + str(i) + "'.")
            if ink.shape[0] > 70:
                result = ink[:70, : ]
            else:
                result = np.zeros((70, 3))
                result[:ink.shape[0], :] = ink

            images.append(img)
            inputs.append(result)
    class_name = file_name.split('%2F')[-1].split('.')[0]
    class_names.append(class_name)


#inputs = np.array(inputs, dtype=np.float16)

#np.save('../quickdraw_data/strokes_data_100', inputs)
#np.save('../quickdraw_data/strokes_data_100_classes', class_names)

image_data = np.load(os.path.join(dir_path,'image_data_100.npy'))

for j in range(0,10):
    images.append(image_data[j,:])

show_images(images,2)