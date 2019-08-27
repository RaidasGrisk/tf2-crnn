import tensorflow as tf


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

