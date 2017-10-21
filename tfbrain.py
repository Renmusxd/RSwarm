from brain import Brain, ToyBrain
import tensorflow as tf
import numpy
import os


class TFBrain(Brain):
    QCopies = 100

    SESS = None
    WRITER = None
    MERGER = None
    SESS_HOLDERS = 0

    # https://stats.stackexchange.com/questions/200006/q-learning-with-neural-network-as-function-approximation/200146
    # https://stats.stackexchange.com/questions/126994/questions-about-q-learning-using-neural-networks
    def __init__(self, name, ninputs, nactions, hshapes=list((25,5)), gamma=0.99, directory="save", rewardbuffer=None):
        # Make a single session if not inherited
        super().__init__(name, ninputs, nactions,
                         directory=directory, rewardbuffer=rewardbuffer)
        self.inherited_sess = False

        self.name = name
        self.directory = directory

        if TFBrain.SESS_HOLDERS == 0:
            tf.reset_default_graph()
        TFBrain.SESS_HOLDERS += 1

        # Put all vars into name scope
        with tf.name_scope(name):
            mainshape = [ninputs]+hshapes+[nactions]

            # Make some models with input/output variables
            self.state_in, self.Qout, self.qnetvars = self.makeqnetwork(mainshape)
            self.next_state, self.dualQout, self.dualnetvars = self.makeqnetwork(mainshape)


            # Copy each var to dual
            self.copy_to_dual = [tf.assign(dualnetvar, qnetvar) for qnetvar, dualnetvar in
                                 zip(self.qnetvars, self.dualnetvars)]

            # Then combine them together to get our final Q-values.
            self.chosen_actions = tf.argmax(self.Qout, 1)
            self.chosen_Q = tf.reduce_sum(tf.multiply(self.Qout,
                                                      tf.one_hot(self.chosen_actions, nactions, dtype=tf.float32)),
                                          axis=1)

            # Below we obtain the loss by taking the sum of squares difference
            # between the target and prediction Q values.
            self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

            self.actions_onehot = tf.one_hot(self.actions, nactions, dtype=tf.float32)

            # Q of chosen actions
            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

            # Q(s,a) = R + y * max_a'( Q(s',a') )
            self.targetQ = self.rewards + gamma * tf.reduce_max(self.dualQout, 1)

            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0005)
            self.updateModel = self.trainer.minimize(self.loss)

            self.saver = tf.train.Saver()

    def makeqnetwork(self, shape):
        """
        Construct graph
        :param inputshape:
        :param noinit:
        :return: input placeholder, output layer, list of variables
        """
        # Build brain model
        state_in = tf.placeholder(shape=[None, shape[0]], dtype=tf.float32)
        layer = state_in
        variables = []

        print(shape)

        for i in range(0, len(shape)-2):
            W = tf.Variable(tf.random_normal([shape[i],shape[i+1]]))
            b = tf.Variable(tf.random_normal([shape[i+1]]))
            layer = tf.nn.relu(tf.add(tf.matmul(layer,W),b))
            variables.append(W)
            variables.append(b)

        # Make output layer without relu
        W = tf.Variable(tf.random_normal([shape[-2],shape[-1]]))
        b = tf.Variable(tf.random_normal([shape[-1]]))
        layer = tf.add(tf.matmul(layer, W), b)
        variables.append(W)
        variables.append(b)

        return state_in, layer, variables

    def get_checkpoint(self):
        return os.path.join(self.directory, self.name if self.name.endswith('.ckpt') else self.name + '.ckpt')

    def has_checkpoint(self):
        return os.path.exists(self.get_checkpoint())

    def loadcheckpoint(self):
        if self.has_checkpoint():
            self.saver.restore(TFBrain.SESS, self.get_checkpoint())

    def think(self, inputs):
        ids = list(inputs.keys())
        acts = TFBrain.SESS.run(self.chosen_actions,
                                feed_dict={self.state_in: [inputs[entityid] for entityid in ids]})
        return {entityid: act for entityid, act in zip(ids, acts)}

    def train(self, niters=1000, batch=64):
        """
        Train the brain for a bit based in rewards previously provided
        :param niters: number of training iterations
        :param batch: batch size
        :return:
        """
        print("Buffer size: {}".format(len(self.buffer)))
        training_gen = self.buffer.get_batch_gen(batchsize=batch, niters=int(niters/TFBrain.QCopies))
        for i in range(TFBrain.QCopies):
            self.trainbatch(training_gen)

    def trainbatch(self, gen):
        # Make dual graph identical to primary
        TFBrain.SESS.run(self.copy_to_dual)

        # Train primary
        for inputs, actions, rewards, newinputs in gen:
            feed_dict = {self.state_in: inputs,
                         self.rewards: rewards,
                         self.actions: actions,
                         self.next_state: newinputs}
            _ = TFBrain.SESS.run([self.updateModel], feed_dict=feed_dict)

    def startup(self):
        super().startup()
        if TFBrain.SESS is None:
            TFBrain.SESS = tf.Session()
            # Logs
            TFBrain.WRITER = tf.summary.FileWriter("output", TFBrain.SESS.graph)

            init = tf.global_variables_initializer()
            TFBrain.SESS.run(init)
        self.loadcheckpoint()


    def save(self):
        """
        Saves variables to directory
        """
        print("Saving {}... ".format(self.name), end='')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.saver.save(TFBrain.SESS, os.path.join(self.directory,self.name+'ckpt'))
        print("Done!")

    def cleanup(self):
        super().cleanup()
        self.save()
        TFBrain.SESS_HOLDERS -= 1
        if TFBrain.SESS_HOLDERS == 0:
            TFBrain.SESS.close()
            TFBrain.WRITER.close()

    def print_diag(self, sample_in):
        qout, dualqout = TFBrain.SESS.run([self.Qout, self.dualQout],
                                          feed_dict={self.state_in: [sample_in],
                                                     self.next_state: [sample_in]})
        print("In:   ", formatarray(sample_in))
        print("Q:    ", formatarray(qout[0]))


def formatarray(array):
    return " ".join("{:5.5f}".format(f) for f in array)
