import numpy as np
import cv2

# the last char is used as a replacement for all the other chars not in the list
char_dict = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"
params = {'SEQ_LENGTH': 47,
          'INPUT_SIZE': [200, 32],
          'NUM_CLASSES': len(char_dict)}


def decode_to_text(char_dict, decoded_out):
    return ''.join([char_dict[i] for i in decoded_out])


def sparse_tuple_from(sequences):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def preprocess_input_image(image, height=params['INPUT_SIZE'][1], width=params['INPUT_SIZE'][0]):

    # fix height
    scale_rate = height / image.shape[0]
    new_width = int(scale_rate * image.shape[1])
    new_width = width if new_width > width else new_width
    image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_LINEAR)

    # fix width
    r, c = np.shape(image)
    if c > width:
        ratio = float(width) / c
        image = cv2.resize(image, (width, int(32 * ratio)))
    else:
        width_pad = width - image.shape[1]
        image = np.pad(image, pad_width=[(0, 0), (0, width_pad)], mode='constant', constant_values=0)

    # add dims
    image = image[:, :, np.newaxis]

    return image


def data_generator(batches=1,
                   batch_size=2,
                   epochs=1,
                   char_dict=char_dict,
                   data_path='D:/data/mnt/ramdisk/max/90kDICT32px/'):

    x_batch = []
    y_batch = []
    for _ in range(epochs):
        with open(data_path + 'annotation_train.txt') as fp:
            for _ in range(batches * batch_size):
                image_path = fp.readline().replace('\n', '').split(' ')[0]

                # get x
                image = cv2.imread(data_path + image_path.replace('./', ''), 0)
                if image is None:
                    continue
                x = preprocess_input_image(image)

                # get y
                y = image_path.split('_')[1]
                y = [char_dict.index(i) if i in char_dict else len(char_dict)-1 for i in y]
                y = y  # + [len(char_dict)-1]

                x_batch.append(x)
                y_batch.append(y)

                if len(y_batch) == batch_size:
                    yield np.array(x_batch).astype(np.float32), np.array(y_batch)
                    x_batch = []
                    y_batch = []


# for x, y in data_generator(batches=2, batch_size=2, epochs=1):
#     print(x.shape, y)