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
print([decode_to_text(char_dict, [j for j in i if j != 0]) for i in decoded])


