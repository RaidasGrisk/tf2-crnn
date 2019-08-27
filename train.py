import tensorflow as tf
import numpy as np
from crnn_model import CRNN
from utils import preprocess_input_image, params, char_dict, decode_to_text, data_generator, sparse_tuple_from

# options
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(edgeitems=30, linewidth=100000)

# init
iter = 0
model = CRNN(num_classes=params['NUM_CLASSES'], training=True)
# model.build(input_shape=(2, 32, 200, 1))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipnorm=5)
# [print(i.name, i.shape) for i in model.trainable_variables]

# training
# dataset: https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth
# please check the data_generator in utils for path to the dataset
# the training set containts 7224612 images / 32 = 225769 batches
for x_batch, y_batch in data_generator(batches=225769, batch_size=32, epochs=1):

    indices, values, dense_shape = sparse_tuple_from(y_batch)
    y_batch_sparse = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)

    with tf.GradientTape() as tape:
        logits, raw_pred, rnn_out = model(x_batch)
        loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y_batch_sparse,
                                             logits=rnn_out,
                                             label_length=[len(i) for i in y_batch],
                                             logit_length=[47]*len(y_batch),
                                             blank_index=62))

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if iter % 100 == 0:

        model.save_weights('checkpoints/model_default')
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits.numpy().transpose((1, 0, 2)),
                                                     sequence_length=[47]*len(y_batch),
                                                     merge_repeated=True)
        decoded = tf.sparse.to_dense(decoded[0]).numpy()
        print(loss.numpy().round(1), [decode_to_text(char_dict, [j for j in i if j != 0]) for i in decoded])

    iter += 1
