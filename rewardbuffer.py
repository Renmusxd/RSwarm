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

class HappyBuffer(object):
    def __init__(self, name, inputsize, directory='save',buffersize=100000):
        self.lezbuffer = RewardBuffer(name + 'lez', inputsize, directory=directory,buffersize=int(buffersize/2))
        self.gtzbuffer = RewardBuffer(name + 'gtz', inputsize, directory=directory,buffersize=int(buffersize/2))

    def reward(self, inputs, actions, rewards, newinputs):
        posents = list(filter(lambda eid: rewards[eid] > 0, rewards.keys()))
        nezents = list(filter(lambda eid: rewards[eid] <= 0, rewards.keys()))

        pinputs, pactions, prewards, pnewinputs = filterdictstokeys(posents, inputs, actions, rewards, newinputs)
        ninputs, nactions, nrewards, nnewinputs = filterdictstokeys(nezents, inputs, actions, rewards, newinputs)

        self.gtzbuffer.reward(pinputs, pactions, prewards, pnewinputs)
        self.lezbuffer.reward(ninputs, nactions, nrewards, nnewinputs)


    def get_batch_gen(self, batchsize, niters):
        """
        Make a generator which provides batches of items
        :param batchsize: size of batch
        :param niters: number of batches to produce
        :return:
        """
        # Array of all (input, action, reward)
        def gen():
            # Choose and yield sets of results
            NARRS = 4
            for i in range(niters):
                total = len(self)
                ngtz = clamp(1, int(batchsize * len(self.gtzbuffer)/total), batchsize-1)
                nlez = clamp(1, int(batchsize * len(self.gtzbuffer)/total), batchsize-1)

                gtzchoices = numpy.random.choice(len(self.gtzbuffer),ngtz)
                lezchoices = numpy.random.choice(len(self.lezbuffer),nlez)

                gtzvals = self.gtzbuffer.states[gtzchoices], self.gtzbuffer.actions[gtzchoices], \
                          self.gtzbuffer.rewards[gtzchoices], self.gtzbuffer.nextstates[gtzchoices]

                lezvals = self.lezbuffer.states[lezchoices], self.lezbuffer.actions[lezchoices], \
                          self.lezbuffer.rewards[lezchoices], self.lezbuffer.nextstates[lezchoices]

                yield tuple(numpy.concatenate([gtzvals[i],  lezvals[i]]) for i in range(NARRS))
        if len(self.gtzbuffer) > 0 and len(self.lezbuffer) > 0:
            return gen()
        elif len(self.gtzbuffer) > 0:
            return self.gtzbuffer.get_batch_gen(batchsize, niters)
        else:
            return self.lezbuffer.get_batch_gen(batchsize, niters)

    def clear(self):
        self.lezbuffer.clear()
        self.gtzbuffer.clear()

    def save(self):
        self.lezbuffer.save()
        self.gtzbuffer.save()

    def load(self):
        self.lezbuffer.load()
        self.gtzbuffer.load()

    def __len__(self):
        return len(self.lezbuffer) + len(self.gtzbuffer)


def clamp(atleast, x, atmost):
    return max(atleast, min(x, atmost))

def filterdictstokeys(keys, *dicts):
    newdicts = []
    for d in dicts:
        newdicts.append({key: d[key] for key in keys})
    return tuple(newdicts)
