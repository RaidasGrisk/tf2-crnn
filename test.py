import tensorflow as tf
import numpy as np
import cv2
import os
from crnn_model import CRNN
from utils import preprocess_input_image, params, char_dict, decode_to_text

# model
model = CRNN(num_classes=params['NUM_CLASSES'], training=False)
model.load_weights('checkpoints/model_default')

# input single img
x = cv2.imread('test_images/test.jpg', 0)
x = preprocess_input_image(x)
x = x[np.newaxis, :, :, :].astype(np.float32)

# input test_images
x = []
for img_dir in os.listdir('test_images'):
    print('/test_images/' + img_dir)
    img = cv2.imread('test_images/{}'.format(img_dir), 0)
    img = preprocess_input_image(img)
    x.append(img)
x = np.array(x).astype(np.float32)

# predict
logits, raw_pred, rnn_out = model(x)
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)),
                                             sequence_length=[params['SEQ_LENGTH']] * x.shape[0],
                                             merge_repeated=True)
decoded = tf.sparse.to_dense(decoded[0]).numpy()

# at this point decoded array contains indices of chars for every word [WORD, CHAR_IDS]
# an id of 0 defines the first char in char_dict. Also, because tf.sparse.to_dense returns zero-padded array
# 0 does not mean anything at all if it is at the end of the array.
# the best solution to this would be to use 0 as a blank index.
# I have not done this, so when decoding the traling 0s in every word vec are trimmed
print([decode_to_text(char_dict, [char for char in np.trim_zeros(word, 'b')]) for word in decoded])

