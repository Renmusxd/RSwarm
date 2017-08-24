from brain import Brain, ToyBrain
import tensorflow as tf
import numpy
import random

import os
from collections import deque


class TFBrain(Brain):

    def __init__(self, name, ninputs, nactions, hshapes=list((10,10)), directory='save', gamma=0.99):
        super().__init__(name, ninputs, nactions, directory)
        self.eps = 0.2
        self.buffer = RewardBuffer(ninputs)
        self.randombrain = ToyBrain(ninputs,nactions,directory)  # For exploration
        self.tensorbrain = TensorflowModel(self.name,ninputs,nactions,hshapes,self.directory,gamma)

        self.reward_cycle = 0

    def think(self, inputs):
        """
        Provides actions for inputs
        :param inputs: dictionary of id:input to think about
        :return: dictionary of id:action
        """
        # If epsilon then explore
        if random.uniform(0,1) < self.eps:
            return self.randombrain.think(inputs)
        else:
            return self.tensorbrain.feedforward(inputs)

    def reward(self, inputs, actions, rewards, newinputs):
        """
        Rewards last actions using Q learning approach
        :param inputs: dictionary of id:[inputs]
        :param actions: dictionary of id:[actions]
        :param rewards: dictionary of id:reward
        :param inputs: dictionary of id:[inputs]
        """
        self.reward_cycle += sum(rewards.values())
        self.buffer.reward(inputs,actions,rewards,newinputs)

    def train(self, niters=1000, batch=64):
        """
        Train the brain for a bit based in rewards previously provided
        :param niters: number of training iterations
        :param batch: batch size
        :return:
        """
        print("Reward for this cycle: {}".format(self.reward_cycle))
        self.reward_cycle = 0
        training_gen = self.buffer.get_batch_gen(batchsize=batch, niters=niters)
        self.tensorbrain.trainbatch(training_gen)

    def startup(self):
        self.randombrain.startup()
        self.tensorbrain.startup()

    def cleanup(self):
        self.randombrain.cleanup()
        self.tensorbrain.cleanup()

    def print_diag(self, sample_in):
        self.tensorbrain.print_diag(sample_in)


class RewardBuffer:
    def __init__(self, inputsize, buffersize=100000):
        """
        :param buffersize:
        """
        self.states = numpy.ndarray(shape=(buffersize,inputsize), dtype=float)
        self.actions = numpy.ndarray(shape=(buffersize,), dtype=int)
        self.rewards = numpy.ndarray(shape=(buffersize,), dtype=float)
        self.nextstates = numpy.ndarray(shape=(buffersize,inputsize), dtype=float)
        self.buffersize = buffersize
        self.size = 0
        self.head = 0

    def reward(self,inputs,actions,rewards,newinputs):
        """
        Provide dicts of {id:item}
        :param inputs:
        :param actions:
        :param rewards:
        :param newinputs:
        """
        for entityid in inputs.keys():
            entityin, entityact = inputs[entityid], actions[entityid]
            entityrew, entitynewin = rewards[entityid], newinputs[entityid]

            self.states[self.head, :] = entityin
            self.actions[self.head] = entityact
            self.rewards[self.head] = entityrew
            self.nextstates[self.head, :] = entitynewin

            self.head = (self.head + 1) % self.buffersize
            self.size = min(self.size+1, self.buffersize)

    def get_batch_gen(self,batchsize,niters):
        """
        Make a generator which provides batches of items
        :param batchsize: size of batch
        :param niters: number of batches to produce
        :return:
        """
        # Array of all (input, action, reward)
        def gen():
            # Choose and yield sets of results
            for i in range(niters):
                choices = numpy.random.choice(self.size,batchsize)
                yield self.states[choices], self.actions[choices], \
                      self.rewards[choices], self.nextstates[choices]
        return gen()

    def clear(self):
        self.size = 0
        self.head = 0

    def __len__(self):
        return self.size


class TensorflowModel:
    """
    This is the actual tensorflow model which is used in the above brain.
    Can be swapped out easily while maintaining the reward training code
    from the TFBrain.
    """
    SESS = None
    WRITER = None
    # MERGER = None
    SESS_HOLDERS = 0

    # https://stats.stackexchange.com/questions/200006/q-learning-with-neural-network-as-function-approximation/200146
    # https://stats.stackexchange.com/questions/126994/questions-about-q-learning-using-neural-networks
    def __init__(self,name,ninputs,nactions,hshapes,directory,gamma=0.99):

        # Make a single session if not inherited
        self.inherited_sess = False

        self.name = name
        self.directory = directory

        if TensorflowModel.SESS_HOLDERS == 0:
            # Clear the Tensorflow graph.
            tf.reset_default_graph()

        TensorflowModel.SESS_HOLDERS += 1

        mainshape = [ninputs]+hshapes+[nactions]
        vashape = ([nactions,nactions],[nactions,nactions])
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

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        self.actions_onehot = tf.one_hot(self.actions, nactions, dtype=tf.float32)

        # Q of chosen actions
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        # Q(s,a) = R + y * max_a'( Q(s',a') )
        self.targetQ = self.reward + gamma * tf.reduce_max(self.dualQout, 1)

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

    def feedforward(self, inputs):
        ids = list(inputs.keys())
        acts = TensorflowModel.SESS.run(self.chosen_actions,
                                        feed_dict={self.state_in: [inputs[entityid] for entityid in ids]})
        return {entityid: act for entityid, act in zip(ids, acts)}

    def trainbatch(self, gen):
        # Make dual graph identical to primary
        TensorflowModel.SESS.run(self.copy_to_dual)

        # Train primary
        for inputs, actions, rewards, newinputs in gen:
            feed_dict = {self.state_in: inputs,
                         self.reward: rewards,
                         self.actions: actions,
                         self.next_state: newinputs}
            _ = TensorflowModel.SESS.run([self.updateModel], feed_dict=feed_dict)
        self.save()

    def startup(self):
        if TensorflowModel.SESS is None:
            init = tf.global_variables_initializer()
            TensorflowModel.SESS = tf.Session()
            # Logs
            TensorflowModel.WRITER = tf.summary.FileWriter("output", TensorflowModel.SESS.graph)
            TensorflowModel.SESS.run(init)

    def loadormakeinits(self, shape, valueadvantageshape=None):
        """
        Loads variables from directory or makes initializers
        :param shape: shape of network (shape[0] = ninputs)
        :param valueadvantageshape: shape of (v,a)
        :return: varinits, (v-a_varinits or None)
        """
        savename = os.path.join(self.directory,self.name if self.name.endswith('.npz') else self.name+'.npz')
        if os.path.exists(savename):
            print("Loading... ", end='')
            loaded = numpy.load(savename)
            loaded_mat = numpy.array([loaded[a] for a in loaded])
            varinits = loaded_mat[0][0]
            if len(loaded_mat) > 1:
                v_varinits = loaded_mat[0][1]
                a_varinits = loaded_mat[0][2]
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

    def save(self):
        """
        Saves variables to directory
        """
        print("Saving... ", end='')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        qnetvarvals = TensorflowModel.SESS.run(self.qnetvars)
        vvarvals = TensorflowModel.SESS.run(self.vvars)
        avarvals = TensorflowModel.SESS.run(self.avars)
        numpy.savez_compressed(os.path.join(self.directory,self.name), [qnetvarvals,vvarvals,avarvals])
        print("Done!")

    def cleanup(self):
        self.save()
        TensorflowModel.SESS_HOLDERS -= 1
        if TensorflowModel.SESS_HOLDERS == 0:
            TensorflowModel.SESS.close()
            TensorflowModel.WRITER.close()

    def print_diag(self, sample_in):
        qout, dualqout = TensorflowModel.SESS.run([self.Qout, self.dualQout],
                                                  feed_dict={self.state_in: [sample_in],
                                                             self.next_state: [sample_in]})
        print("In:   ", formatarray(sample_in))
        print("Q:    ", formatarray(qout[0]))


def formatarray(array):
    return " ".join("{:5.5f}".format(f) for f in array)
