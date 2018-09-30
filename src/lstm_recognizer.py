import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import data_handler_lstm as data
import numpy as np

hm_epochs = 100
n_classes = 6
batch_size = 
chunk_size = 561
n_chunks = 47
rnn_size = 128
max_len = 47

inputs, labels = data.prepare_data(data.content, data.labels, max_len)
test_inputs, test_labels = data.prepare_data(data.test_content,
                                             data.test_labels, max_len)
x = tf.placeholder(tf.float32, [None, n_chunks, chunk_size])
y = tf.placeholder(tf.float32)


dropout_rate = 0.6


def conv1d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool1d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


def cnn_rnn(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 1, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 1, 32, 64])),
               'out': tf.Variable(tf.random_normal([rnn_size, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'out': tf.Variable(tf.random_normal([n_classes]))}
    cnn_out = []
    x = tf.reshape(x, shape=[-1, n_chunks, chunk_size, 1])
    conv1 = tf.nn.relu(conv1d(x,
                       weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool1d(conv1)
    conv2 = tf.nn.relu(conv1d(conv1,
                       weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool1d(conv2)
    x = conv2
    dims = conv2.get_shape()
    number_of_elements = dims[2:].num_elements()
    print(number_of_elements, dims)
    x = tf.reshape(x, [-1, 12, number_of_elements])
    # x = tf.reshape(x, [-1, n_chunks, chunk_size])
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, number_of_elements])
    x = tf.split(x, 12, 0)
    # print(np.array(x).shape)

    def rnn_cell(size):
        cell = rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell,
                                             output_keep_prob=dropout_rate)
        return cell
    cells = [rnn_cell(rnn_size) for _ in range(1)]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # fc = tf.nn.dropout(fc, keep_rate)

    output = tf.add(tf.matmul(outputs[-1], weights['out']), biases['out'])
    return output


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    return output


def train_neural_network(x):
    prediction = cnn_rnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,
                          y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(inputs):
                epoch_x = np.array(inputs[i:i+batch_size])
                epoch_y = np.array(labels[i:i+batch_size])
                _, c = sess.run([optimizer, cost],
                                feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch, 'completed out of',
                  hm_epochs, 'loss:', epoch_loss)
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: np.array(test_inputs),
              y: np.array(test_labels)}))


train_neural_network(x)
