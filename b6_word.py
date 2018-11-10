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


batch_size = 5600
no_epochs = 10
lr = 0.01

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

def twolayer_model(x):
    with tf.variable_scope("foo" ):
        word_vectors = tf.contrib.layers.embed_sequence(
          x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

        word_list = tf.unstack(word_vectors, axis=1)

        cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
        cell = tf.contrib.rnn.MultiRNNCell([cell] * 2)
        _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

        logits = tf.layers.dense(encoding[-1], MAX_LABEL, activation=None)

        # TODO: make sure that dropout is not applied on testing
    return logits, word_list

def lstm_model(x):
    with tf.variable_scope("foo" ):
        word_vectors = tf.contrib.layers.embed_sequence(
          x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

        word_list = tf.unstack(word_vectors, axis=1)

        cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

        logits = tf.layers.dense(encoding[0], MAX_LABEL, activation=None)
    return logits, word_list

def vanilla_model(x):
    with tf.variable_scope("foo" ):
        word_vectors = tf.contrib.layers.embed_sequence(
          x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

        word_list = tf.unstack(word_vectors, axis=1)

        cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
        _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

        logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
    return logits, word_list

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

def chooseTrainOp(useClipping, entropy):
    if useClipping:
        minimizer = tf.train.AdamOptimizer()
        grads_and_vars = minimizer.compute_gradients(entropy)

        # Gradient clipping
        grad_clipping = tf.constant(2.0, name="grad_clipping")
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            clipped_grad = tf.clip_by_value(grad, -grad_clipping, grad_clipping)
            clipped_grads_and_vars.append((clipped_grad, var))

            # Gradient updates
        train_op = minimizer.apply_gradients(clipped_grads_and_vars)
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(entropy)
    return train_op


def main():
    global n_words
    #x_train is a list of ids corresponding to each word in the paragraph.
    x_train, y_train, x_test, y_test, n_words = data_read_words()
    # print('y_train:', y_train.shape)
    # Create the model
    loss = {}
    test_accs = {}
    #word_vectors is the vector representation of each id
    tf.reset_default_graph()
    x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
    y_ = tf.placeholder(tf.int64)

    logits, word_list = twolayer_model(x)
    entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
    train_op = chooseTrainOp(True, entropy)
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_ ), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    p = 0
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
            _, loss_  = sess.run([train_op, entropy], {x: x_train, y_: y_train})
        loss[p].append(loss_)
        acc = sess.run([accuracy], {x: x_test, y_: y_test})
        test_accs[p].append(acc[0])
        if e%1 == 0:
            print('epoch: %d, entropy: %g'%(e, loss[p][e]), 'accuracy:', test_accs[p][e])

    print(loss)
    print(test_accs)
    for k,v in loss.items():
        plt.figure("Entropy Vs Epochs")
        plt.plot(range(no_epochs), v, label= str(k) + ' dropout rate')
        plt.xlabel(str(no_epochs) + ' Epochs')
        plt.ylabel('Entropy')
    plt.legend()
    plt.savefig('b6'+'lossvsepochs.png')

    for k,v in test_accs.items():
        test_fig = plt.figure("Accuracy Vs Epochs")
        plt.plot(range(no_epochs), v, label= str(k) + ' dropout rate')
        plt.xlabel(str(no_epochs) + ' Epochs')
        plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('b6'+'accvsepochs.png')


if __name__ == '__main__':
    main()
