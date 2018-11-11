import numpy as np
import pandas
import tensorflow as tf
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
FILTER_SHAPE1 = [20, 256]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2
MAX_LABEL = 15
batch_size = 128
no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def char_cnn_model(x, p):

    input_layer = tf.reshape(
      tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])

    with tf.variable_scope('CNN_Layer1'):
        conv1 = tf.layers.conv2d(
            input_layer,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE1,
            padding='VALID',
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')
        conv2 = tf.layers.conv2d(
            pool1,
            filters=N_FILTERS,
            kernel_size=FILTER_SHAPE2,
            padding='VALID',
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(
            conv2,
            pool_size=POOLING_WINDOW,
            strides=POOLING_STRIDE,
            padding='SAME')

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

    logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)
    logits = tf.layers.dropout(logits, rate=p, training=True)
    return input_layer, logits


def read_data_chars():

    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[1])
            y_train.append(int(row[0]))

    with open('test_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[1])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)


    char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
    x_train = np.array(list(char_processor.fit_transform(x_train)))
    x_test = np.array(list(char_processor.transform(x_test)))
    y_train = y_train.values
    y_test = y_test.values

    return x_train, y_train, x_test, y_test


def main():

    x_train, y_train, x_test, y_test = read_data_chars()

    print(len(x_train))
    print(len(x_test))

    loss = {}
    test_accs = {}
    #word_vectors is the vector representation of each id
    for p in [0.0, 0.2, 0.4, 0.6, 0.8]:
        tf.reset_default_graph()
        x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
        y_ = tf.placeholder(tf.int64)
        prob = tf.placeholder(tf.float32)
        input_layer, logits = char_cnn_model(x,prob)
        entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_ ), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

      # training
        loss[p] = []
        test_accs[p] = []
        idx = np.arange(x_train.shape[0])
        NUM_INPUT = x_train.shape[0]
        repetition_in_one_epoch = int(NUM_INPUT / batch_size)
        for e in range(no_epochs):
            np.random.shuffle(idx)
            x_train, y_train = x_train[idx], y_train[idx]
            start = -1 * batch_size
            end = 0
            for k in range(repetition_in_one_epoch):
                start += batch_size
                end += batch_size
                if end > NUM_INPUT:
                    end = NUM_INPUT
                input_layer_, _, loss_  = sess.run([input_layer, train_op, entropy], {x: x_train[start:end], y_: y_train[start:end], prob:p})
            loss[p].append(loss_)
            acc = sess.run([accuracy], {x: x_test, y_: y_test, prob:0.0 })
            test_accs[p].append(acc[0])
            if e%10 == 0:
                print('epoch: %d, entropy: %g'%(e, loss[p][e]), 'accuracy:', test_accs[p][e])

    print(loss)
    print(test_accs)
    for k,v in loss.items():
        plt.figure("Entropy Vs Epochs")
        plt.plot(range(no_epochs), v, label= str(k) + ' dropout rate')
        plt.xlabel(str(no_epochs) + ' Epochs')
        plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('lossvsepochs.png')

    for k,v in test_accs.items():
        test_fig = plt.figure("Accuracy Vs Epochs")
        plt.plot(range(no_epochs), v, label= str(k) + ' dropout rate')
        plt.xlabel(str(no_epochs) + ' Epochs')
        plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accvsepochs.png')

if __name__ == '__main__':
    main()
