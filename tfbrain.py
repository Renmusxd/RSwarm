from brain import Brain, ToyBrain
import tensorflow as tf
import numpy
import random

from collections import deque


class TFBrain(Brain):

    def __init__(self, ninputs, nactions, hshapes=list((10,10)), gamma=0.99):
        super().__init__(ninputs, nactions)
        self.eps = 0.2
        self.buffer = RewardBuffer()
        self.randombrain = ToyBrain(ninputs,nactions)  # For exploration
        self.tensorbrain = TensorflowModel(ninputs,nactions,hshapes,gamma)

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

    def train(self, niters=10000, batch=64):
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
    """
    Reward buffer is used to store and produce event recall and
    delayed reward calculations.
    """

    def __init__(self, buffersize=1000000,removebatch=10000):
        """
        :param gamma: Q learning gamma factor (0 -> short term, 1 -> long term)
        :param buffersize: Number of entries to store, may be briefly surpassed during deletion
        :param removebatch: number to remove in event of overflow (approx)
        """
        self.experiences = {}
        self.buffersize = buffersize
        self.removebatch = removebatch
        self.size = 0

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

            if entityid not in self.experiences:
                self.experiences[entityid] = deque()
            self.experiences[entityid].append((entityin,entityact,entityrew,entitynewin))

            self.size += 1
        if self.size > self.buffersize:
            while self.size > self.buffersize-self.removebatch and self.size > 0:
                for entityid in list(self.experiences.keys()):
                    rews = self.experiences[entityid]
                    rews.popleft()
                    if len(rews) == 0:
                        self.experiences.pop(entityid)
                    self.size -= 1

    def get_batch_gen(self,batchsize,niters):
        """
        Make a generator which provides batches of items
        :param batchsize: size of batch
        :param niters: number of batches to produce
        :return:
        """
        # Array of all (input, action, reward)
        def gen():
            # Make flat numpy arrays all inputs, actions, and rewards for each entity
            flatgen = (item for sublist in self.experiences.values() for item in sublist)
            flatinputs,flatactions,flatrewards,flatnewinputs = map(numpy.array, zip(*flatgen))
            nitems = self.size
            # Choose and yield sets of results
            for i in range(niters):
                choices = numpy.random.choice(nitems,batchsize)
                yield flatinputs[choices], flatactions[choices], flatrewards[choices], flatnewinputs[choices]
        return gen()

    def clear(self):
        self.experiences = {}
        self.size = 0

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
    def __init__(self,ninputs,nactions,hshapes,gamma=0.99):

        # Make a single session if not inherited
        self.inherited_sess = False

        if TensorflowModel.SESS_HOLDERS == 0:
            # Clear the Tensorflow graph.
            tf.reset_default_graph()

        TensorflowModel.SESS_HOLDERS += 1

        # Make some models with input/output variables
        self.state_in, self.Qout, self.qnetvars = self.makeqnetwork([ninputs]+hshapes+[nactions])
        self.next_state, self.dual_Qout, self.dualnetvars = self.makeqnetwork([ninputs] + hshapes + [nactions])

        self.copy_to_dual = [tf.assign(dualnetvar,qnetvar) for qnetvar, dualnetvar in zip(self.qnetvars,self.dualnetvars)]

        # Then combine them together to get our final Q-values.
        self.chosen_actions = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

        self.actions_onehot = tf.one_hot(self.actions, nactions, dtype=tf.float32)

        # Q of chosen actions
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        # Q(s,a) = R + y * max_a'( Q(s',a') )
        self.targetQ = self.reward + gamma * tf.reduce_max(self.dual_Qout,1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

    def makeqnetwork(self,shape):
        # Build brain model
        state_in = tf.placeholder(shape=[None, shape[0]], dtype=tf.float32)
        layer = state_in
        variables = []
        for i in range(1,len(shape)-1):
            W = tf.Variable(tf.random_normal([shape[i-1], shape[i]]))
            b = tf.Variable(tf.random_normal([shape[i]]))
            layer = tf.nn.relu(tf.add(tf.matmul(layer,W),b))
            variables.append(W)
            variables.append(b)
        W = tf.Variable(tf.random_normal([shape[-2], shape[-1]]))
        b = tf.Variable(tf.random_normal([shape[-1]]))
        Qout = tf.add(tf.matmul(layer,W),b)
        variables.append(W)
        variables.append(b)
        return state_in, Qout, variables

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

    def startup(self):
        if TensorflowModel.SESS is None:
            init = tf.global_variables_initializer()
            TensorflowModel.SESS = tf.Session()
            # Logs
            TensorflowModel.WRITER = tf.summary.FileWriter("output", TensorflowModel.SESS.graph)
            TensorflowModel.SESS.run(init)

    def cleanup(self):
        TensorflowModel.SESS.close()
        TensorflowModel.WRITER.close()

    def print_diag(self, sample_in):
        qout, dualqout = TensorflowModel.SESS.run([self.Qout, self.dual_Qout],
                                                  feed_dict={self.state_in: [sample_in],
                                                             self.next_state: [sample_in]})
        print("In:   ", formatarray(sample_in))
        print("Q:    ", formatarray(qout[0]))


def formatarray(array):
    return " ".join("{:5.5f}".format(f) for f in array)
