import numpy


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
                yield self.states[choices], self.actions[choices], self.rewards[choices], self.nextstates[choices]
        return gen()

    def clear(self):
        self.size = 0
        self.head = 0

    def __len__(self):
        return self.size