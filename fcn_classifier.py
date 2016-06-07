import tensorflow as tf
import os

class FCNClassifier:
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def __init__(self, xlen, hidden_sizes, ylen, train_keep_probs, learning_rate, ckpt_path):

        num_h_layers = len(hidden_sizes)
        self.learning_rate = learning_rate
        self.train_keep_probs = train_keep_probs
        self.keep_probs = tf.placeholder(shape = (num_h_layers), dtype = tf.float32)
        self.ckpt_path = ckpt_path
        self.predict_keep_probs = [1 for i in range(num_h_layers)]

        self.x = tf.placeholder(tf.float32, [None, xlen])

        w = [self.weight_variable((xlen, hidden_sizes[0]))]

        for i in range(1, num_h_layers):
            w.append(self.weight_variable((hidden_sizes[i-1], hidden_sizes[i])))
        w.append(self.weight_variable((hidden_sizes[-1], ylen)))

        b = []
        for layer_size in hidden_sizes:
            b.append(self.bias_variable((1, layer_size)))
        b.append(self.bias_variable((1, ylen)))

        yl = []
        yl.append(tf.nn.relu(tf.matmul(self.x, w[0]) + b[0]))
        dropout_layer = tf.nn.dropout(yl[-1], self.keep_probs[0])
        yl.append(dropout_layer)
        for i in range(num_h_layers-1):
            yl.append(tf.nn.relu(tf.matmul(yl[-1], w[i+1]) + b[i+1]))
            dropout_layer = tf.nn.dropout(yl[-1], self.keep_probs[i+1])
            yl.append(dropout_layer)


        yl.append(tf.matmul(yl[-1], w[-1]) + b[-1])
        self.y = yl[-1]

        self.y_ = tf.placeholder(tf.float32, [None, ylen])

        loss = tf.reduce_mean(tf.square(self.y_-self.y))

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print "Loaded checkpoints %s" % (ckpt.model_checkpoint_path)
        else:
            print "Couldn't load checkpoints"


    def train(self, xdata, ydata):
        self.sess.run(self.train_step, feed_dict={self.x: xdata, self.y_: ydata, self.keep_probs: self.train_keep_probs})

    def get_accuracy(self, xdata, ydata):
        return self.sess.run(self.accuracy, feed_dict={self.x: xdata, self.y_: ydata, self.keep_probs: self.predict_keep_probs})

    def save(self):
        self.saver.save(self.sess, self.ckpt_path + '/model.ckpt')

    def predict(self, xdata):
        return selfsess.run(self.y, feed_dict={self.x: xdata, self.keep_probs: self.predict_keep_probs})
