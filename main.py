"""

https://github.com/Belval/CRNN/blob/master/CRNN/crnn.py

https://github.com/solivr/tf-crnn/blob/master/tf_crnn/model.py
https://github.com/MaybeShewill-CV/CRNN_Tensorflow/blob/master/crnn_model/crnn_net.py

https://github.com/tensorflow/tensorflow/issues/28070
https://github.com/sbillburg/CRNN-with-STN

-----------------------------
https://towardsdatascience.com/build-a-handwritten-text-recognition-system-using-tensorflow-2326a3487cd5



TODO:


"""


import tensorflow as tf
import numpy as np
import cv2
import os

np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=30, linewidth=100000)


def sparse_tuple_from(sequences, dtype=np.int32):
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
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_to_text(char_dict, decoded_out):
    return ''.join([char_dict[i] for i in decoded_out])


class CRNN(tf.keras.Model):

    """
    cnn in:         [BATCH, HEIGHT, WIDTH, CHANNELS]    ->  [x, 32, 200, 1]
    cnn out:        [BATCH, CHANNELS, TIME, FILTERS]    ->  [x, 1, 47, 512]
    rnn in:         [BATCH, TIME, FILTERS]              ->  [x, 47, 512]
    rnn out:
        raw:        [BATCH, TIME, FILTERS]              ->  [x, 47, 512]
        logits:     [BATCH, TIME, CHAR-LEN]             ->  [x, 47, 63]
        raw_pred:   [BATCH, TIME]                       ->  [x, 47]
        rnn_out:    [TIME, BATCH, CHAR-LEN]             ->  [47, x, 63]
    """

    def __init__(self, num_classes, training):
        super(CRNN, self).__init__()

        # cnn
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu, dtype='float32')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)

        self.conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.bn3 = tf.keras.layers.BatchNormalization(trainable=training)

        self.conv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])

        self.conv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.bn5 = tf.keras.layers.BatchNormalization(trainable=training)

        self.conv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)
        self.pool6 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])

        self.conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), padding="valid", activation=tf.nn.relu)

        # rnn
        self.lstm_fw_cell_1 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.lstm_bw_cell_1 = tf.keras.layers.LSTM(256, go_backwards=True, return_sequences=True)
        self.birnn1 = tf.keras.layers.Bidirectional(layer=self.lstm_fw_cell_1, backward_layer=self.lstm_bw_cell_1)

        self.lstm_fw_cell_2 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.lstm_bw_cell_2 = tf.keras.layers.LSTM(256, go_backwards=True, return_sequences=True)
        self.birnn2 = tf.keras.layers.Bidirectional(layer=self.lstm_fw_cell_2, backward_layer=self.lstm_bw_cell_2)

        self.dense = tf.keras.layers.Dense(num_classes)  # number of classes + 1 blank char

    def call(self, input):

        # (3, 3, 1, 64)
        # (?, 32, 200, 64) -> (?, 32, 200, 64) -> (?, 16, 100, 64)
        x = self.conv1(input)
        x = self.pool1(x)

        # (3, 3, 64, 64)
        # (?, 16, 100, 64) -> (?, 16, 100, 64) -> (?, 8, 50, 64)
        x = self.conv2(x)
        x = self.pool2(x)

        # (3, 3, 64, 256)
        # (?, 8, 50, 64) -> (?, 8, 50, 256) -> (?, 8, 50, 256)
        x = self.conv3(x)
        x = self.bn3(tf.cast(x, dtype=tf.float32))  # bn does not support this witough a cast

        # (3, 3, 256, 256)
        # (?, 8, 50, 256) -> (?, 8, 50, 256) -> (?, 4, 49, 256)
        x = self.conv4(x)
        x = self.pool4(x)

        # (3, 3, 256, 512)
        # (?, 4, 49, 256) -> (?, 4, 49, 512) -> (?, 4, 49, 512)
        x = self.conv5(x)
        x = self.bn5(tf.cast(x, dtype=tf.float32))  # bn does not support this witough a cast

        # (3, 3, 512, 512)
        # (?, 4, 49, 512) -> (?, 4, 49, 512) -> (?, 2, 48, 512)
        x = self.conv6(x)
        x = self.pool6(x)

        # (2, 2, 512, 512)
        # (?, 2, 48, 512) -> (?, 1, 47, 512)
        x = self.conv7(x)

        # (512, 1024) (256, 1024) & (512, 1024) (256, 1024)
        # (?, 1, 47, 512) -> (?, 47, 512) -> (?, 47, 512) & (?, 47, 512)
        x = tf.reshape(x, [-1, x.shape[2], x.shape[3]])  # [BATCH, TIME, FILTERS]
        x = self.birnn1(x)
        x = self.birnn2(x)

        # (512, 63)
        # (?, 47, 512) -> (?, 47, 63)
        logits = self.dense(x)

        raw_pred = tf.argmax(tf.nn.softmax(logits), axis=2)
        rnn_out = tf.transpose(logits, [1, 0, 2])

        return logits, raw_pred, rnn_out

    #
    # def call(self, input):
    #
    #     # conv1
    #     x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)(input)
    #     print(x.shape)
    #     x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
    #     print(x.shape)
    #
    #     # conv2
    #     x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)(x)
    #     print(x.shape)
    #     x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)(x)
    #     print(x.shape)
    #     # conv3
    #     # activation function after batch_norm?
    #     x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)(x)
    #     print(x.shape)
    #     x = tf.keras.layers.BatchNormalization(trainable=True)(tf.cast(x, dtype=tf.float32))
    #     print(x.shape)
    #     # conv4
    #     x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)(x)
    #     print(x.shape)
    #     x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])(x)
    #     print(x.shape)
    #     # conv5
    #     # activation function after batch_norm?
    #     x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)(x)
    #     print(x.shape)
    #     x = tf.keras.layers.BatchNormalization(trainable=True)(tf.cast(x, dtype=tf.float32))
    #     print(x.shape)
    #     # conv6
    #     x = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation=tf.nn.relu)(x)
    #     print(x.shape)
    #     x = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 1])(x)
    #     print(x.shape)
    #     # conv7
    #     # + batch_norm?
    #     x = tf.keras.layers.Conv2D(filters=512, kernel_size=(2, 2), padding="valid", activation=tf.nn.relu)(x)
    #     print(x.shape)
    #
    #     # bilstm
    #     # B S 512
    #     x = tf.reshape(x, [tf.shape(x)[0], -1, 512])
    #     print(x.shape)
    #
    #     lstm_fw_cell_1 = tf.keras.layers.LSTM(256, return_sequences=True)
    #     lstm_bw_cell_1 = tf.keras.layers.LSTM(256, go_backwards=True, return_sequences=True)
    #     x = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_1, backward_layer=lstm_bw_cell_1)(x)
    #     print(x.shape)
    #
    #     # lstm_fw_cell_1 = tf.keras.layers.LSTM(256)
    #     # lstm_bw_cell_1 = tf.keras.layers.LSTM(256, go_backwards=True)
    #     # x = tf.keras.layers.Bidirectional(layer=lstm_fw_cell_1, backward_layer=lstm_bw_cell_1)(x)
    #     # output = tf.concat(x, 2)
    #
    #     logits = tf.keras.layers.Dense(63+1)(x)
    #     raw_pred = tf.argmax(tf.nn.softmax(logits), axis=1)
    #     rnn_out = tf.transpose(logits, [1, 0, 2])
    #
    #     return logits, raw_pred, rnn_out


# init
char_dict = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-"
params = {'SEQ_LENGTH': 47,
          'INPUT_SIZE': [100, 32],
          'NUM_CLASSES': len(char_dict)}

model = CRNN(num_classes=params['NUM_CLASSES'], training=True)
model.build(input_shape=(2, 32, 200, 1))
print('\n')
[print(i.name, i.shape) for i in model.trainable_variables]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5)  # SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
input_width = 200

# data
train_path = 'data//'
content = os.listdir(train_path)

x_batch = []
y_batch = []

for i in content:

    if len(y_batch) == 10:
        break

    image = cv2.imread(train_path + i.replace('./', ''), 0)

    # cv2.imshow('image', image.astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # fix height
    new_heigth = 32
    scale_rate = new_heigth / image.shape[0]
    new_width = int(scale_rate * image.shape[1])
    new_width = new_width if new_width > params['INPUT_SIZE'][0] else params['INPUT_SIZE'][0]
    image = cv2.resize(image, (new_width, new_heigth), interpolation=cv2.INTER_LINEAR)

    # fix width
    r, c = np.shape(image)
    if c > input_width:
        c = input_width
        ratio = float(input_width) / c
        image = cv2.resize(image, (input_width, int(32 * ratio)))
    else:
        final_arr = np.zeros((32, input_width))
        ratio = 32 / r
        im_arr_resized = cv2.resize(image, (int(c * ratio), 32))
        final_arr[:, 0:min(input_width, np.shape(im_arr_resized)[1])] = im_arr_resized[:, 0:input_width]
        image = final_arr.copy()

    if len(image.shape) == 3:
        image = image[np.newaxis, :, :, np.newaxis].astype(np.float32)
    else:
        image = image[np.newaxis, :, :, np.newaxis].astype(np.float32)

    y = [char_dict.index(x) if x in char_dict else '1' for x in i.split('/')[-1].split('_')[1]]
    y = y + [params['NUM_CLASSES']]
    y = np.array(y).reshape(-1)
    x_batch.append(image)
    y_batch.append(y)


[print(i.shape) for i in x_batch]

x_batch = np.squeeze(np.array(x_batch), 1)
y_batch = np.array(y_batch)

# y sparse
indices, values, dense_shape = sparse_tuple_from(y_batch)
y_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

# Training loop
for iter in range(2000):

    with tf.GradientTape() as tape:
        logits, raw_pred, rnn_out = model(x_batch)
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_sparse,  # [batch, max_length]
                                             logits=rnn_out,
                                             label_length=[len([j for j in i if j != 63]) for i in y_batch],
                                             logit_length=[47]*len(y_batch),  # [len(i) for i in y_batch]
                                             blank_index=62))  # definately 63 not 64

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if iter % 20 == 0:

        logits_reshaped = logits.numpy().transpose((1, 0, 2))
        # ctc_beam_search_decoder
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_reshaped,
                                                     sequence_length=[47]*len(y_batch),
                                                     merge_repeated=True)
        decoded = tf.sparse.to_dense(decoded[0]).numpy()
        print(loss.numpy().round(1),
              decoded.tolist(),
              [decode_to_text(char_dict, [j for j in i if j != 0]) for i in decoded])


# ----------- #
# model.save_weights('model_10_samples')
# model.load_weights('model_10_samples')


# debug
# if there are two datapoints in a batch, both y_ will contain both datapoins.
# not sure why, but thats a bug, most likely
decode_to_text(char_dict, [i for i in decoded[5] if i != 0])
decode_to_text(char_dict, [i for i in logits[5, :, :].numpy().argmax(axis=1)])

