import tensorflow as tf
import math
import cv2


class Evaluator:
    def __init__(self, _category_bundle):
        self._category_bundle = _category_bundle
        self._num_images_per_class = {}
        self._correct_per_class = {}
        for categories in self._category_bundle:
            for category in categories:
                self._num_images_per_class[category] = 0
                self._correct_per_class[category] = 0

    def add(self, correct, path):
        for categories in self._category_bundle:
            for category in categories:
                if path.find('/' + category + '/') != -1:
                    self._num_images_per_class[category] += 1
                    if correct:
                        self._correct_per_class[category] += 1
                    return

    def print_scores(self):
        all_total = 0
        all_correct = 0
        scores = []
        print('--------------------------')
        print('* Scores')
        for label, categories in zip(range(999), self._category_bundle):
            for category in categories:
                total = self._num_images_per_class[category]
                correct = self._correct_per_class[category]
                all_total += total
                all_correct += correct
                score = correct / total
                scores.append(score)
                print(' - %02d. ' % label, category.rjust(30), ' : %.03f' % score, '= %05d' % correct, '/ %05d' % total)
        print('--------------------------')
        all_score = all_correct / all_total
        print('Total Score : %.03f'%all_score, '= %05d' % all_correct, '/ %05d' % all_total)
        return all_score, scores


class LeNet:
    def __init__(self, sess):
        self.sess = sess

        # Network
        self.conv1_filter_size = 5
        self.conv1_dim = 32
        self.conv2_filter_size = 5
        self.conv2_dim = 32
        self.fc1_dim = 1000
        self.fc2_dim = 500
        self.decay_steps = 20

        self.keep_prob = tf.placeholder(tf.float32)

        self.standards = [0.9, 0.95, 0.98, 1.0]
        self.learning_rates = [0.0001, 0.0001, 0.00005, 0.00002]

    def get_learning_rate(self, accuracy):
        for standard, learning_rate in zip(self.standards, self.learning_rates):
            if accuracy <= standard:
                return learning_rate

    def set_train_buffer(self, buffer):
        self._train_buffer = buffer

    def set_test_buffer(self, buffer):
        self._test_buffer = buffer

    def test(self, imshow=False):
        print('\n\nTest Start!')
        buffer = self._test_buffer
        if self.shape != buffer.shape() or self.num_label != buffer.num_label():
            raise Exception('Shape of train buffer and test buffer is not same')

        input = self._input
        ground_truth = self._ground_truth
        logits = self._logits

        batch_size = 100
        batch_count = int(math.ceil(buffer.size() / batch_size))

        evaluator = Evaluator(buffer.get_category_bundle())
        for idx in range(batch_count):
            batch_x, batch_y, paths = buffer.next_batch(batch_size)

            predictions, answers = self.sess.run([tf.argmax(logits, 1), tf.argmax(ground_truth, 1)],
                                       feed_dict={input: batch_x, ground_truth: batch_y, self.keep_prob:1.0})
            for path, prediction, answer in zip(paths, predictions, answers):
                correct = prediction == answer
                evaluator.add(correct, path)
                if imshow:
                    print('Path : ', path)
                    image = cv2.imread(path)
                    print('* ', str(correct).rjust(5), ' - Prediction vs Answer :', prediction, answer)
                    cv2.imshow('image', image)
                    if cv2.waitKey(0)==ord('c'):
                        print('Skip imshow!')
                        imshow = False
        all_score, scores = evaluator.print_scores()
        print('Test Total Accuracy : ', all_score)

    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model.ckpt")

    def build(self):
        self.shape = self._train_buffer.shape()
        self.num_label = self._train_buffer.num_label()
        shape = self.shape
        num_label = self.num_label

        self._input = tf.placeholder(tf.float32, [None, shape[0]*shape[1]*shape[2]])
        self._ground_truth = tf.placeholder(tf.float32, [None, num_label])
        self._logits = self._build_network(self._input, shape, num_label)

    def train(self):
        input = self._input
        ground_truth = self._ground_truth
        logits = self._logits

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth, logits=logits))
        learning_rate = tf.placeholder(tf.float32)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        correct_prediction = tf.equal(tf.argmax(ground_truth, 1), tf.argmax(logits, 1))
        # accuracy_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        next_learning_rate = self.learning_rates[0]

        for epoch in range(99999):
            batch_size = 100
            batch_count = int(math.ceil(self._train_buffer.size() / batch_size))

            self._train_buffer.reset()
            for idx in range(batch_count):
                batch_x, batch_y, _ = self._train_buffer.next_batch(batch_size)
                self.sess.run(train_step,
                              feed_dict={input: batch_x, ground_truth: batch_y, learning_rate: next_learning_rate,
                                         self.keep_prob: 0.9})
            self._train_buffer.reset()

            evaluator = Evaluator(self._train_buffer.get_category_bundle())
            for idx in range(batch_count):
                batch_x, batch_y, paths = self._train_buffer.next_batch(batch_size)
                prediction = self.sess.run(correct_prediction,
                                           feed_dict={input: batch_x, ground_truth: batch_y, self.keep_prob: 1.0})
                for path, prediction in zip(paths, prediction):
                    evaluator.add(prediction, path)

            print('\n\nEpoch %02d' % epoch)
            all_score, scores = evaluator.print_scores()
            print('- Graph saved at %s' % saver.save(self.sess, "./model.ckpt"))

            next_learning_rate = self.get_learning_rate(all_score)
            print('next_learning_rate : ', next_learning_rate)

            if all_score >= 0.99:
                print('Training Complete!')
                break

    def _build_network(self, input, shape, num_label):
        input_image = tf.reshape(input, [-1, shape[0], shape[1], shape[2]])

        initializer = tf.contrib.layers.xavier_initializer()

        # Convolution layer 1
        filter_size = self.conv1_filter_size
        conv1_w = tf.Variable(initializer([filter_size, filter_size, shape[2], self.conv1_dim]), name='conv1_w')
        conv1_b = tf.Variable(initializer([self.conv1_dim]), name='conv1_b')
        conv1_conv = tf.nn.conv2d(input_image, conv1_w, strides=[1, 1, 1, 1], padding='VALID')+conv1_b
        conv1_relu = tf.nn.relu(conv1_conv)

        # Pooling layer 1
        pool1_max_pool = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # Convolution layer 2
        filter_size = self.conv2_filter_size
        conv2_w = tf.Variable(initializer([filter_size, filter_size, self.conv1_dim, self.conv2_dim]), name='conv2_w')
        conv2_b = tf.Variable(initializer([self.conv2_dim]), name='conv2_b')
        conv2_conv = tf.nn.conv2d(pool1_max_pool, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        conv2_relu = tf.nn.relu(conv2_conv)

        # Pooling layer 2
        pool2_max_pool = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool2_max_pool_dim = int(pool2_max_pool.shape[1] * pool2_max_pool.shape[2] * pool2_max_pool.shape[3])
        flat_pool2_max_pool = tf.reshape(pool2_max_pool, [-1, pool2_max_pool_dim])

        # Fully connected layer 1
        fc1_w = tf.Variable(initializer([pool2_max_pool_dim, self.fc1_dim]), name='fc1_w')
        fc1_b = tf.Variable(initializer([self.fc1_dim]), name='fc1_b')
        fc1_fc = tf.matmul(flat_pool2_max_pool, fc1_w) + fc1_b

        # Fully connected layer 2
        fc2_w = tf.Variable(initializer([self.fc1_dim, self.fc2_dim]), name='fc2_w')
        fc2_b = tf.Variable(initializer([self.fc2_dim]), name='fc2_b')
        fc2_fc = tf.matmul(fc1_fc, fc2_w) + fc2_b

        # Fully connected layer 3
        fc3_w = tf.Variable(initializer([self.fc2_dim, num_label]), name='fc3_w')
        fc3_b = tf.Variable(initializer([num_label]), name='fc3_b')

        fc3_fc = tf.matmul(fc2_fc, fc3_w) + fc3_b

        return fc3_fc

    def _build_network2(self, input, shape, num_label):
        keep_prob = self.keep_prob

        input_image = tf.reshape(input, [-1, shape[0], shape[1], shape[2]])

        initializer = tf.contrib.layers.xavier_initializer()

        # Convolution layer 1
        filter_size = self.conv1_filter_size
        conv1_w = tf.Variable(initializer([filter_size, filter_size, shape[2], self.conv1_dim]), name='conv1_w')
        conv1_b = tf.Variable(initializer([self.conv1_dim]), name='conv1_b')
        conv1_conv = tf.nn.conv2d(input_image, conv1_w, strides=[1, 1, 1, 1], padding='VALID')+conv1_b

        # Pooling layer 1
        pool1_max_pool = tf.nn.max_pool(conv1_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # LRN ayer 1
        lrn1 = tf.nn.lrn(pool1_max_pool, 4, bias=1.0, alpha=0.0001 / 9.0, beta=0.75, name='lrn1')

        # Convolution layer 2
        filter_size = self.conv2_filter_size
        conv2_w = tf.Variable(initializer([filter_size, filter_size, self.conv1_dim, self.conv2_dim]), name='conv2_w')
        conv2_b = tf.Variable(initializer([self.conv2_dim]), name='conv2_b')
        conv2_conv = tf.nn.conv2d(lrn1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

        # Pooling layer 2
        pool2_max_pool = tf.nn.max_pool(conv2_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        # LRN ayer 1
        lrn2 = tf.nn.lrn(pool2_max_pool, 4, bias=1.0, alpha=0.0001 / 9.0, beta=0.75, name='lrn2')

        # Reshape
        lrn2_dim = int(lrn2.shape[1] * lrn2.shape[2] * lrn2.shape[3])
        flat_lrn2 = tf.reshape(lrn2, [-1, lrn2_dim])

        # Fully connected layer 1
        fc1_w = tf.Variable(initializer([lrn2_dim, self.fc1_dim]), name='fc1_w')
        fc1_b = tf.Variable(initializer([self.fc1_dim]), name='fc1_b')
        fc1_fc = tf.matmul(flat_lrn2, fc1_w) + fc1_b

        # ReLU
        conv1_relu = tf.nn.relu(fc1_fc)

        # Dropout
        dropout_conv1_relu = tf.nn.dropout(conv1_relu, keep_prob)

        # Fully connected layer 2
        fc2_w = tf.Variable(initializer([self.fc1_dim, self.fc2_dim]), name='fc2_w')
        fc2_b = tf.Variable(initializer([self.fc2_dim]), name='fc2_b')
        fc2_fc = tf.matmul(dropout_conv1_relu, fc2_w) + fc2_b

        # ReLU
        conv2_relu = tf.nn.relu(fc2_fc)

        # Dropout
        dropout_conv2_relu = tf.nn.dropout(conv2_relu, keep_prob)

        # Fully connected layer 3
        fc3_w = tf.Variable(initializer([self.fc2_dim, num_label]), name='fc3_w')
        fc3_b = tf.Variable(initializer([num_label]), name='fc3_b')
        fc3_fc = tf.matmul(dropout_conv2_relu, fc3_w) + fc3_b

        return fc3_fc
