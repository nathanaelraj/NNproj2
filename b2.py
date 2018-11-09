import numpy as np
import pandas
import tensorflow as tf
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_DOCUMENT_LENGTH = 100
HIDDEN_SIZE = 20
MAX_LABEL = 15
EMBEDDING_SIZE = 20
N_FILTERS = 10
FILTER_SHAPE1 = [20, 20]
FILTER_SHAPE2 = [20, 1]
POOLING_WINDOW = 4
POOLING_STRIDE = 2

batch_size = 2800
no_epochs = 100
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def rnn_model(x):

    word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

    word_list = tf.unstack(word_vectors, axis=1)

    cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
    _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

    logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

    return logits, word_list

def word_cnn_model(x):

    word_vectors = tf.contrib.layers.embed_sequence(
      x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    input_layer = tf.reshape(
      word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1])
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

    return logits, word_vectors


def data_read_words():

    x_train, y_train, x_test, y_test = [], [], [], []

    with open('train_medium.csv', encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_train.append(row[2])
            y_train.append(int(row[0]))

    with open("test_medium.csv", encoding='utf-8') as filex:
        reader = csv.reader(filex)
        for row in reader:
            x_test.append(row[2])
            y_test.append(int(row[0]))

    x_train = pandas.Series(x_train)
    y_train = pandas.Series(y_train)
    x_test = pandas.Series(x_test)
    y_test = pandas.Series(y_test)
    y_train = y_train.values
    y_test = y_test.values

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

    x_transform_train = vocab_processor.fit_transform(x_train)
    x_transform_test = vocab_processor.transform(x_test)
    x_train = np.array(list(x_transform_train))
    x_test = np.array(list(x_transform_test))

    no_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % no_words)

    return x_train, y_train, x_test, y_test, no_words

def main():
    global n_words
    #x_train is a list of ids corresponding to each word in the paragraph.
    x_train, y_train, x_test, y_test, n_words = data_read_words()
    # print('y_train:', y_train.shape)
    # Create the model
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    #word_vectors is the vector representation of each id
    logits, word_vectors = word_cnn_model(x)
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_ ), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

  # training
    loss = []
    test_accs = []
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
            word_vectors_, _, loss_  = sess.run([word_vectors, train_op, entropy], {x: x_train, y_: y_train})

        loss.append(loss_)
        acc = sess.run([accuracy], {x: x_test, y_: y_test})
        test_accs.append(acc[0])

        if e%10 == 0:
            print('epoch: %d, entropy: %g'%(e, loss[e]), 'accuracy:', test_accs[e])


    sess.close()
    plt.figure("Entropy Vs Epochs")
    plt.plot(range(no_epochs), loss)
    plt.xlabel(str(no_epochs) + ' Epochs')
    plt.ylabel('Entropy')
    plt.savefig('lossvsepochs.png')

    plt.figure("Accuracy Vs Epochs")
    plt.plot(range(no_epochs), test_accs)
    plt.xlabel(str(no_epochs) + ' Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('accvsepochs.png')

if __name__ == '__main__':
    main()
