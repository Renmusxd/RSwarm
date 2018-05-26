from brain import Brain
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import rnn_common
import os


class RNNBrain(Brain):
    DUALCOPYFREQ = 1000

    SESS = None
    WRITER = None
    MERGER = None
    SESS_HOLDERS = 0

    def __init__(self, name, ninputs, nactions, hshapes=list((25,10)), gamma=0.9, directory="save", rewardbuffer=None):
        # Make a single session if not inherited
        super().__init__(name, ninputs, nactions,
                         directory=directory, rewardbuffer=rewardbuffer)
        self.inherited_sess = False

        self.name = name
        self.directory = directory

        if RNNBrain.SESS_HOLDERS == 0:
            tf.reset_default_graph()
        RNNBrain.SESS_HOLDERS += 1

        # Put all vars into name scope
        with tf.name_scope(name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            # Make some models with input/output variables
            rnnshape = [10]
            ffnshape = [10, 10]

            dual_training_tuple, dual_inference_tuple, self.dual_qnetvars = self.makeqnetwork(ninputs, rnnshape, ffnshape, nactions,
                                                                                              scope_name="dual_rnn")
            self.dual_training_input, dual_sequence_lengths, self.dual_training_Qout = dual_training_tuple
            dual_inference_input, dual_inference_hidden_state, dual_inference_layer, dual_inference_newstate = dual_inference_tuple

            # Dual and normal share inputs, but dual takes one extra time step to get the last entry.
            main_sequence_lengths = dual_sequence_lengths - 1
            training_tuple, inference_tuple, self.qnetvars = self.makeqnetwork(ninputs, rnnshape, ffnshape, nactions,
                                                                               training_input=self.dual_training_input,
                                                                               training_sequence_lengths=main_sequence_lengths,
                                                                               inference_input=dual_inference_input,
                                                                               inference_hidden_state=dual_inference_hidden_state,
                                                                               scope_name="rnn")

            self.training_input, self.training_sequence_lengths, self.training_Qout = training_tuple
            self.inference_input, self.inference_state, self.inference_Qout, self.inference_newstate = inference_tuple

            # Take processed time series and turn into Q value prediction.
            self.training_Qprobs, self.training_Qactions = self.maptoactions(self.training_Qout)
            self.inference_Qprobs, self.inference_Qactions = self.maptoactions(self.inference_Qout)
            self.dual_training_Qprobs, self.dual_training_Qactions = self.maptoactions(self.dual_training_Qout)

            # Copy each var to dual
            self.copy_to_dual = [tf.assign(dualnetvar, qnetvar) for qnetvar, dualnetvar in
                                 zip(self.qnetvars, self.dual_qnetvars)]

            # Then combine them together to get our final Q-values.
            self.prob_chosen_Q = tf.reduce_sum(tf.multiply(self.training_Qactions,
                                                           tf.one_hot(self.training_Qactions, nactions,
                                                                      dtype=tf.float32)),
                                               axis=1)
            # Below we obtain the loss by taking the sum of squares difference
            # between the target and prediction Q values.
            self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

            self.actions_onehot = tf.one_hot(self.actions, nactions, dtype=tf.float32)
            # Q of chosen actions
            self.Q = tf.reduce_sum(tf.multiply(self.training_Qout, self.actions_onehot), axis=1)

            # Q(s,a) = R + y * max_a'( Q(s',a') )
            self.targetQ = self.rewards + gamma * tf.reduce_max(self.dual_training_Qout, 1)
            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=0.0002)
            self.updateModel = self.trainer.minimize(self.loss, global_step=self.global_step)

            self.saver = tf.train.Saver()
            with tf.name_scope('summary'):
                self.losssum = tf.summary.scalar('loss', self.loss)

                self.rewardsumvar = tf.placeholder(shape=(), name="episode_reward", dtype=tf.float32)
                self.rewardsum = tf.summary.scalar('reward', self.rewardsumvar)

    def makeqnetwork(self, input_size, rnnshape, ffnshape, num_actions,
                     training_input=None, training_sequence_lengths=None,
                     inference_input=None, inference_hidden_state=None,
                     scope_name="RNN"):
        """
        Construct graph.
        :return: input placeholder, output layer, list of variables
        """
        # Build brain model
        with tf.name_scope(scope_name + '/') as ns:
            rnn_layers = [tf.nn.rnn_cell.GRUCell(units) for units in rnnshape]
            multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

            if training_input is not None and training_sequence_lengths is not None:
                state_in = training_input
                sequence_lengths = training_sequence_lengths
            else:
                state_in = tf.placeholder(shape=[None, None, input_size], dtype=tf.float32)
                sequence_lengths = tf.placeholder(shape=[None], dtype=tf.int32)

            outputs, hidden = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                                inputs=state_in,
                                                sequence_length=sequence_lengths,
                                                dtype=tf.float32)
            layer = rnn_common.select_last_activations(outputs, sequence_lengths)

            if inference_input is not None and inference_hidden_state is not None:
                inference_in = inference_input
                inference_state = inference_hidden_state
            else:
                inference_in = tf.placeholder(shape=[None, input_size], dtype=tf.float32)
                inference_state = multi_rnn_cell.zero_state(tf.shape(inference_in)[0], dtype=tf.float32)

            inference_output, inference_hidden = multi_rnn_cell(inference_in, inference_state)
            inference_layer = inference_output

            for i, units in enumerate(ffnshape):
                layer = tf.layers.dense(layer, units,
                                        activation=tf.nn.relu, reuse=None, name="ffn_{}".format(i))
            for i, units in enumerate(ffnshape):
                inference_layer = tf.layers.dense(inference_layer, units,
                                                  activation=tf.nn.relu, reuse=True, name="ffn_{}".format(i))

            # Make output layer without relu
            layer = tf.layers.dense(layer, num_actions,
                                    activation=None, reuse=None, name="ffn_last")
            inference_layer = tf.layers.dense(inference_layer, num_actions,
                                              activation=None, reuse=True, name="ffn_last")

            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=ns)

        return (state_in, sequence_lengths, layer), (inference_in, inference_state, inference_layer, inference_hidden), variables

    def maptoactions(self, logits):
        qprobs = tf.nn.softmax(logits, 1)
        qout = tf.reshape(tf.multinomial(tf.log(qprobs), 1), [-1])
        return qprobs, qout

    def get_checkpoint(self):
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)
        return os.path.join(self.directory, self.name)

    def has_checkpoint(self):
        return tf.train.checkpoint_exists(self.get_checkpoint())

    def loadcheckpoint(self):
        if self.has_checkpoint():
            print("Loading checkpoint... ", end="")
            self.saver.restore(RNNBrain.SESS, self.get_checkpoint())
            print("Done!")
        else:
            print("No checkpoint found")

    def think(self, inputs, memory):
        if len(inputs) == 0:
            return {}

        ids = list(inputs.keys())
        # Here we can choose either prob_chosen_actions or chosen_actions
        # TODO return hidden states too
        acts, newstates = RNNBrain.SESS.run([self.inference_Qout, self.inference_newstate],
                                            feed_dict={self.inference_input: [inputs[entityid] for entityid in ids],
                                                       self.inference_state: [memory[entityid] for entityid in ids]})
        eacts = {entityid: act for entityid, act in zip(ids, acts)}
        emems = {entityid: mem for entityid, mem in zip(ids, newstates)}
        return eacts, emems

    def debug(self,debuginput, debugmemory):
        actprobs = RNNBrain.SESS.run(self.inference_Qprobs, feed_dict={self.inference_input: [debuginput],
                                                                       self.inference_state: [debugmemory]})
        return actprobs[0]

    def train(self, iters, batch, totreward=None):
        """
        Train the brain for a bit based in rewards previously provided
        :param niters: number of training iterations
        :param batch: batch size
        :return:
        """
        if totreward is not None:
            rewardsum, global_step = RNNBrain.SESS.run([self.rewardsum, self.global_step],
                                                       feed_dict={self.rewardsumvar: totreward})
            RNNBrain.WRITER.add_summary(rewardsum, global_step)

        print("Buffer size: {}".format(len(self.buffer)))
        print("Gen niters: {}".format(int(RNNBrain.DUALCOPYFREQ)))
        print("\tncopies: {}".format(int(iters/RNNBrain.DUALCOPYFREQ)))
        for i in range(int(iters/RNNBrain.DUALCOPYFREQ)):
            training_gen = self.buffer.get_batch_gen(batchsize=batch, niters=int(RNNBrain.DUALCOPYFREQ))
            self.trainbatch(training_gen)

    def trainbatch(self, gen):
        # Make dual graph identical to primary
        RNNBrain.SESS.run(self.copy_to_dual)

        # Train primary
        for inputs, seq_lengths, actions, rewards in gen:
            feed_dict = {self.training_input: inputs,
                         self.training_sequence_lengths: seq_lengths,
                         self.rewards: rewards,
                         self.actions: actions}
            _, summary, global_step = RNNBrain.SESS.run([self.updateModel, self.losssum, self.global_step],
                                                       feed_dict=feed_dict)
            RNNBrain.WRITER.add_summary(summary, global_step)

    def startup(self):
        super().startup()
        if RNNBrain.SESS is None:
            RNNBrain.SESS = tf.Session()
            # Logs
            RNNBrain.WRITER = tf.summary.FileWriter("output", RNNBrain.SESS.graph)

            init = tf.global_variables_initializer()
            RNNBrain.SESS.run(init)
        self.loadcheckpoint()

    def save(self):
        """
        Saves variables to directory
        """
        print("Saving {}... ".format(self.name), end='')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.saver.save(RNNBrain.SESS, self.get_checkpoint())
        print("Done!")

    def cleanup(self):
        super().cleanup()
        self.save()
        RNNBrain.SESS_HOLDERS -= 1
        if RNNBrain.SESS_HOLDERS == 0:
            RNNBrain.SESS.close()
            RNNBrain.WRITER.close()

    def print_diag(self, sample_in):
        pass


def formatarray(array):
    return " ".join("{:5.5f}".format(f) for f in array)

if __name__ == "__main__":
    b = RNNBrain('rnnbrain', 10, 5)