import itertools

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Dense, Activation, LSTM
from keras.layers import Reshape, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import SGD

from util.generator import ImageGenerator
from util.generator import LETTERS


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def predict(img_w, max_text_len, weights):
    # Input Parameters
    img_h = 64

    # Network parameters
    conv_filters = 16
    kernel_size = (3, 3)
    pool_size = 2
    time_dense_size = 32
    rnn_size = 512

    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    act = 'relu'
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv1')(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max1')(inner)
    inner = Conv2D(conv_filters, kernel_size, padding='same',
                   activation=act, kernel_initializer='he_normal',
                   name='conv2')(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name='max2')(inner)

    conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters)
    inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(inner)

    # cuts down input size going into RNN:
    inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

    # Two layers of bidirecitonal LSTMs
    lstm_1 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)
    lstm_1b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(
        inner)
    lstm1_merged = add([lstm_1, lstm_1b])
    lstm_2 = LSTM(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
        lstm1_merged)

    # transforms RNN output to character activations:
    inner = Dense(ImageGenerator.get_output_size(), kernel_initializer='he_normal',
                  name='dense2')(concatenate([lstm_2, lstm_2b]))
    y_pred = Activation('softmax', name='softmax')(inner)
    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    # clipnorm seems to speeds up convergence
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    model.load_weights(weights)
    return model


# For a real OCR application, this should be beam search with a dictionary
# and language model.  For this example, best path is sufficient.
def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(LETTERS):
                outstr += LETTERS[c]
        ret.append(outstr)
    return ret


if __name__ == '__main__':
    sess = tf.Session()
    K.set_session(sess)

    tiger_test = ImageGenerator('../../../data/captcha_solver/test', 128, 64, 8, 4)
    tiger_test.build_data()

    model = predict(128, tiger_test.max_text_len,
                    '../../../model/rnnLSTM6-7-8.hdf5')

    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output

    n = 0
    valid = 0

    for inp_value, _ in tiger_test.next_batch():
        bs = inp_value['the_input'].shape[0]
        X_data = inp_value['the_input']
        net_out_value = sess.run(net_out, feed_dict={net_inp: X_data})
        pred_texts = decode_batch(net_out_value)
        labels = inp_value['the_labels']
        texts = []
        for label in labels:
            text = ''.join(list(map(lambda x: LETTERS[int(x)], label)))
            texts.append(text)

        for i in range(bs):
            n += 1
            if pred_texts[i] == texts[i]:
                valid += 1
            print('Predicted: %s\nTrue: %s' % (pred_texts[i], texts[i]))
        if n >= 1500:
            break
    print('Acurancy:{}%'.format(valid/n * 100))
