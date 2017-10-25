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


class PositiveReserveBuffer(RewardBuffer):
    def __init__(self, name, inputsize, maxposratio=0.5, **kwargs):
        super().__init__(name, inputsize, **kwargs)
        self.maxposhead = int(self.buffersize*maxposratio)
        self.poshead = 0
        self.possize = 0

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

            print(self.poshead, self.head, self.possize, self.size)

            if entityrew > 0:
                # Displace old value, replace with new one
                entityin, self.states[self.poshead, :] = self.states[self.poshead, :], entityin
                entityact, self.actions[self.poshead] = self.actions[self.poshead], entityact
                entityrew, self.rewards[self.poshead] = self.rewards[self.poshead], entityrew
                entitynewin, self.nextstates[self.poshead, :] = self.nextstates[self.poshead, :], entitynewin

                # Increase poshead and wrap around if needed
                self.poshead = (self.poshead + 1) % self.maxposhead
                self.possize = min(self.possize+1, self.maxposhead)

                # Make sure head is not inside the possize area
                self.head = max(self.possize, self.head)

            print(self.poshead, self.head, self.possize, self.size)

            if entityrew <= 0:
                self.states[self.head, :] = entityin
                self.actions[self.head] = entityact
                self.rewards[self.head] = entityrew
                self.nextstates[self.head, :] = entitynewin

                # Even if we added to pos, increase head to reflect larger size
                self.head = max((self.head + 1) % self.buffersize, self.possize)
                self.size = min(self.size+1, self.buffersize)

            print(self.poshead, self.head, self.possize, self.size)

def clamp(atleast, x, atmost):
    return max(atleast, min(x, atmost))


def filterdictstokeys(keys, *dicts):
    newdicts = []
    for d in dicts:
        newdicts.append({key: d[key] for key in keys})
    return tuple(newdicts)


if __name__ == '__main__':
    r = PositiveReserveBuffer('TEST',1,buffersize=10)
    r.rewards[:] = 0

    def rew(x):
        r.reward({0: 0}, {0: 0}, {0: x}, {0: 0})
        print(r.rewards)

    rew(-1)
    rew(-1)
    rew(1)
    rew(1)
    rew(-1)
