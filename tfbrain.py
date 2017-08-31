from brain import Brain, ToyBrain
import tensorflow as tf
import numpy
import os


class TFBrain(Brain):
    QCopies = 10

    SESS = None
    WRITER = None
    # MERGER = None
    SESS_HOLDERS = 0

    # https://stats.stackexchange.com/questions/200006/q-learning-with-neural-network-as-function-approximation/200146
    # https://stats.stackexchange.com/questions/126994/questions-about-q-learning-using-neural-networks
    def __init__(self, name, ninputs, nactions, hshapes=list((5, 5)), gamma=0.99, directory="save", rewardbuffer=None):
        # Make a single session if not inherited
        super().__init__(name, ninputs, nactions,
                         directory=directory, rewardbuffer=rewardbuffer)
        self.inherited_sess = False

        self.name = name
        self.directory = directory

        if TFBrain.SESS_HOLDERS == 0:
            # Clear the Tensorflow graph.
            tf.reset_default_graph()
        TFBrain.SESS_HOLDERS += 1

        mainshape = [ninputs]+hshapes
        vshape = (hshapes[-1], nactions)
        ashape = (hshapes[-1], nactions)
        vashape = (vshape, ashape)
        varinits, va_varinits = self.loadormakeinits(mainshape,vashape)

        # Make some models with input/output variables
        self.state_in, self.Qout, self.qnetvars, self.vvars, self.avars = \
            self.makeqnetwork(ninputs, varinits, va_varinits)
        self.next_state, self.dualQout, self.dualnetvars, self.dualvvars, self.dualavars = \
            self.makeqnetwork(ninputs, varinits, va_varinits)

        # Copy each var to dual
        self.copy_to_dual = [tf.assign(dualnetvar, qnetvar) for qnetvar, dualnetvar in
                             zip(self.qnetvars + self.vvars + self.avars,
                                 self.dualnetvars + self.dualvvars + self.dualavars)]

        # Then combine them together to get our final Q-values.
        self.chosen_actions = tf.argmax(self.Qout, 1)
        self.chosen_Q = tf.reduce_sum(tf.multiply(self.Qout,
                                                  tf.one_hot(self.chosen_actions, nactions, dtype=tf.float32)),
                                      axis=1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
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

    def makeqnetwork(self,inputshape,varinits,valueadvantageinits=None):
        """
        Construct graph
        :param inputshape:
        :param varinits:
        :param valueadvantageinits: tuple of (valueinits,advantageinits)
        :return: input placeholder, output layer, list of variables
        """
        # Build brain model
        state_in = tf.placeholder(shape=[None, inputshape], dtype=tf.float32)
        layer = state_in
        variables = []
        v_variables, a_variables = [], []

        for i in range(0, len(varinits)-2, 2):
            W = tf.Variable(varinits[i])
            b = tf.Variable(varinits[i+1])
            layer = tf.nn.relu(tf.add(tf.matmul(layer,W),b))
            variables.append(W)
            variables.append(b)
        W = tf.Variable(varinits[-2])
        b = tf.Variable(varinits[-1])
        layer = tf.add(tf.matmul(layer, W), b)
        variables.append(W)
        variables.append(b)

        if valueadvantageinits is not None:
            valueinits, advantageinits = valueadvantageinits

            # Make value pipeline
            v_layer = layer
            for i in range(0, len(valueinits) - 2, 2):
                W = tf.Variable(valueinits[i])
                b = tf.Variable(valueinits[i + 1])
                v_layer = tf.nn.relu(tf.add(tf.matmul(v_layer, W), b))
                v_variables.append(W)
                v_variables.append(b)
            W = tf.Variable(valueinits[-2])
            b = tf.Variable(valueinits[-1])
            v_layer = tf.add(tf.matmul(v_layer, W), b)
            v_variables.append(W)
            v_variables.append(b)

            # Make advantage pipeline
            a_layer = layer
            for i in range(0, len(advantageinits) - 2, 2):
                W = tf.Variable(advantageinits[i])
                b = tf.Variable(advantageinits[i + 1])
                a_layer = tf.nn.relu(tf.add(tf.matmul(a_layer, W), b))
                a_variables.append(W)
                a_variables.append(b)
            W = tf.Variable(advantageinits[-2])
            b = tf.Variable(advantageinits[-1])
            a_layer = tf.add(tf.matmul(a_layer, W), b)
            a_variables.append(W)
            a_variables.append(b)

            Qout = v_layer + tf.subtract(a_layer, tf.reduce_mean(a_layer, axis=1, keep_dims=True))
        else:
            Qout = layer

        return state_in, Qout, variables, v_variables, a_variables

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
        self.save()

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
            init = tf.global_variables_initializer()
            TFBrain.SESS = tf.Session()
            # Logs
            TFBrain.WRITER = tf.summary.FileWriter("output", TFBrain.SESS.graph)
            TFBrain.SESS.run(init)

    def save(self):
        """
        Saves variables to directory
        """
        print("Saving... ", end='')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        qnetvarvals = TFBrain.SESS.run(self.qnetvars)
        vvarvals = TFBrain.SESS.run(self.vvars)
        avarvals = TFBrain.SESS.run(self.avars)
        numpy.savez_compressed(os.path.join(self.directory,self.name),
                               qnet=qnetvarvals, vvar=vvarvals, avar=avarvals)
        print("Done!")

    def loadormakeinits(self, shape, valueadvantageshape=None):
        """
        Loads variables from directory or makes initializers
        :param shape: shape of network (shape[0] = ninputs)
        :param valueadvantageshape: shape of (v,a)
        :return: varinits, (v-a_varinits or None)
        """
        savename = os.path.join(self.directory, self.name if self.name.endswith('.npz') else self.name+'.npz')
        if os.path.exists(savename):
            print("Loading brain... ", end='')
            loaded = numpy.load(savename)
            varinits = loaded['qnet']
            if 'vvar' in loaded:
                v_varinits = loaded['vvar']
                a_varinits = loaded['avar']
                va_varinits = (v_varinits, a_varinits)
            else:
                va_varinits = None
            print("Done!")
        else:
            # For each of the shape, value, and advantage
            varinits = []
            for i in range(len(shape)-1):
                winit = tf.random_normal([shape[i],shape[i+1]])
                binit = tf.random_normal([shape[i+1]])
                varinits.append(winit)
                varinits.append(binit)

            if valueadvantageshape is None:
                va_varinits = None
            else:
                v_shape, a_shape = valueadvantageshape

                v_varinits = []
                for i in range(len(v_shape)-1):
                    winit = tf.random_normal([v_shape[i],v_shape[i+1]])
                    binit = tf.random_normal([v_shape[i+1]])
                    v_varinits.append(winit)
                    v_varinits.append(binit)
                a_varinits = []
                for i in range(len(v_shape) - 1):
                    winit = tf.random_normal([a_shape[i], a_shape[i + 1]])
                    binit = tf.random_normal([a_shape[i + 1]])
                    a_varinits.append(winit)
                    a_varinits.append(binit)
                va_varinits = (v_varinits, a_varinits)
        return varinits, va_varinits

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
