import tensorflow as tf
import math


class Evaluator:
    []


class LeNet:
    def __init__(self, sess):
        self.sess = sess

        # Network
        self.conv1_filter_size = 5
        self.conv1_dim = 32
        self.conv2_filter_size = 5
        self.conv2_dim = 32
        self.fc1_dim = 500

        # Train
        self.stages = [0.0, 0.5, 0.9, 0.99]
        self.batch_sizes = [200, 100, 50, 25]
        self.learning_rates = [0.001, 0.0001, 0.0001, 0.00003]

    def set_train_buffer(self, buffer):
        self._train_buffer = buffer

    def set_test_buffer(self, buffer):
        self._test_buffer = buffer

    def _get_batch_size_learning_rage(self, accuracy):
        batch_size = 0
        learning_rate = 0
        for idx, stage in zip(range(len(self.stages)), self.stages):
            if accuracy >= stage:
                batch_size = self.batch_sizes[idx]
                learning_rate = self.learning_rates[idx]
        return batch_size, learning_rate

    def train(self):
        self.shape = self._train_buffer.shape()
        self.num_label = self._train_buffer.num_label()
        shape = self.shape
        num_label = self.num_label

        batch_size, learning_rate = self._get_batch_size_learning_rage(0)

        input = tf.placeholder(tf.float32, [None, shape[0]*shape[1]*shape[2]])
        ground_truth = tf.placeholder(tf.float32, [None, num_label])
        logits = self._build_network(input, shape, num_label)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth, logits=logits))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(ground_truth, 1), tf.argmax(logits, 1))
        accuracy_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(99999):
            batch_count = int(math.ceil(self._train_buffer.size() / batch_size))

            for idx in range(batch_count):
                batch_x, batch_y, _ = self._train_buffer.next_batch(batch_size)
                self.sess.run(train_step, feed_dict={input:batch_x, ground_truth:batch_y})

            accuracy_sum = 0
            for idx in range(batch_count):
                batch_x, batch_y, paths = self._train_buffer.next_batch(batch_size)
                # accuracy = self.sess.run(accuracy_step, feed_dict={input: batch_x, ground_truth: batch_y})
                # accuracy_sum += accuracy * len(batch_x)
                prediction = self.sess.run(correct_prediction, feed_dict={input: batch_x, ground_truth: batch_y})



            accuracy = accuracy_sum / self._train_buffer.size()

            print('Epoch %02d' % epoch, ' - Train accuracy : %.03f' % accuracy)
            print('- Graph saved at %s' % saver.save(self.sess, "./model.ckpt"))
            batch_size, learning_rate = self._get_batch_size_learning_rage(accuracy)
            print('- Next batch size : ', batch_size)
            print('- Next learning rate : ', learning_rate)

    def _build_network(self, input, shape, num_label):
        input_image = tf.reshape(input, [-1, shape[0], shape[1], shape[2]])

        initializer = tf.contrib.layers.xavier_initializer

        # Convolution layer 1
        filter_size = self.conv1_filter_size
        conv1_w = tf.get_variable('conv1_W', shape=[filter_size, filter_size, shape[2], self.conv1_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv1_b = tf.get_variable('conv1_b', shape=[self.conv1_dim], initializer=tf.contrib.layers.xavier_initializer())
        conv1_conv = tf.nn.conv2d(input_image, conv1_w, strides=[1, 1, 1, 1], padding='VALID')+conv1_b
        conv1_activation = tf.nn.relu(conv1_conv)

        # Pooling layer 1
        pool1_max_pool = tf.nn.max_pool(conv1_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Convolution layer 2
        filter_size = self.conv2_filter_size
        conv2_w = tf.get_variable('conv2_W', shape=[filter_size, filter_size, self.conv1_dim, self.conv2_dim],
                                  initializer=tf.contrib.layers.xavier_initializer())
        conv2_b = tf.get_variable('conv2_b', shape=[self.conv2_dim], initializer=tf.contrib.layers.xavier_initializer())
        conv2_conv = tf.nn.conv2d(pool1_max_pool, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2_activation = tf.nn.relu(conv2_conv)

        # Pooling layer 2
        pool2_max_pool = tf.nn.max_pool(conv2_activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool2_max_pool_dim = int(pool2_max_pool.shape[1] * pool2_max_pool.shape[2] * pool2_max_pool.shape[3])
        flat_pool2_max_pool = tf.reshape(pool2_max_pool, [-1, pool2_max_pool_dim])

        # Fully connected layer 1
        fc1_w = tf.get_variable('fc1_w', shape=[pool2_max_pool_dim, self.fc1_dim],
                                initializer=tf.contrib.layers.xavier_initializer())
        fc1_b = tf.get_variable('fc1_b', shape=[self.fc1_dim], initializer=tf.contrib.layers.xavier_initializer())
        fc1_fc = tf.matmul(flat_pool2_max_pool, fc1_w) + fc1_b
        fc1_fc_dim = int(fc1_fc.shape[1])

        # Fully connected layer 2
        fc2_w = tf.get_variable('fc2_w', shape=[self.fc1_dim, num_label],
                                initializer=tf.contrib.layers.xavier_initializer())
        fc2_b = tf.get_variable('fc2_b', shape=[num_label], initializer=tf.contrib.layers.xavier_initializer())
        fc2_fc = tf.matmul(fc1_fc, fc2_w) + fc2_b

        return fc2_fc

