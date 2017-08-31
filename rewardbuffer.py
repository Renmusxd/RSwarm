import numpy
import os


class RewardBuffer:
    def __init__(self, name, inputsize,
                 directory='save',buffersize=100000):
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
        self.name = name
        self.directory = directory
        self.dirty = False

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
        self.dirty = True

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
        self.dirty = True

    def save(self):
        if self.dirty:
            print("Saving buffer... ",end='')
            substates = self.states[:self.size]
            subactions = self.actions[:self.size]
            subrewards = self.rewards[:self.size]
            subnext = self.nextstates[:self.size]
            numpy.savez_compressed(os.path.join(self.directory, self.name),
                                   states=substates, actions=subactions,
                                   rewards=subrewards, nexts=subnext)
            print("Done!")
            self.dirty = False

    def load(self):
        savename = os.path.join(self.directory, self.name if self.name.endswith('.npz') else self.name + '.npz')
        if os.path.exists(savename) and not self.dirty:
            print("Loading buffer... ", end='')

            loaded = numpy.load(savename)

            substates = loaded['states']
            subactions = loaded['actions']
            subrewards = loaded['rewards']
            subnext = loaded['nexts']

            self.states[:len(substates)] = substates
            self.actions[:len(subactions)] = subactions
            self.rewards[:len(subrewards)] = subrewards
            self.nextstates[:len(subnext)] = subnext
            self.size = len(substates)
            self.head = self.size-1
            self.dirty = True
            print("Done!")

    def __len__(self):
        return self.size
