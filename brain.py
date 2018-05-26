from rewardbuffer import *

from abc import ABCMeta, abstractmethod
import random


class Brain(metaclass=ABCMeta):
    DEFAULTBATCH = 64
    DEFAULTITERS = 10000

    def __init__(self, name, ninputs, nactions, directory='save', rewardbuffer=None):
        """
        Construct a new brain
        :param ninputs: number of inputs
        :param nactions: number of possible actions
        """
        self.ninputs = ninputs
        self.nactions = nactions
        self.name = name
        self.directory = directory
        if rewardbuffer is None:
            rewardbuffer = RewardBuffer("RB_{}".format(name),ninputs)
        self.buffer = rewardbuffer

    @abstractmethod
    def think(self, inputs, memory):
        """
        Provides actions for inputs
        :param inputs: dictionary of id:input to think about
        :param memory: dictionary of id:memory to use while thinking
        :return: dictionary of id:actions, dictionary of id:memory
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, iters, batch, totreward=None):
        """
        Train the brain for a bit based in rewards previously provided
        :param iters:
        :param batch:
        :return:
        """
        raise NotImplementedError

    def print_diag(self, sample_in):
        pass

    def startup(self):
        """
        If overridden make sure to call super
        """
        self.buffer.load()

    def cleanup(self):
        """
        If overridden make sure to call super
        """
        self.buffer.save()

    def reward(self, inputs, actions, rewards, newinputs):
        """
        Rewards last actions using Q learning approach
        :param inputs: dictionary of id:[inputs]
        :param actions: dictionary of id:[actions]
        :param rewards: dictionary of id:reward
        :param newinputs: dictionary of id:[inputs]
        """
        self.buffer.reward(inputs, actions, rewards, newinputs)

    def default_memory(self):
        """
        Get the default memory for this brain.
        :return: Memory object of type dependent on brain.
        """
        return None

    def debug(self,debuginput,debugmemory):
        return None


class ToyBrain(Brain):
    def __init__(self, name, ninputs, nactions, directory='save', rewardbuffer=None):
        super().__init__(name, ninputs, nactions,
                         directory=directory, rewardbuffer=rewardbuffer)

    def think(self, inputs, memory):
        return {entityid: random.randint(0, self.nactions - 1)
                for entityid in inputs.keys()}, {}

    def train(self, iters=1000000, batch=64, totreward=None):
        pass

    def debug(self, debuginput,debugmemory):
        return None


class CombinedBrain(Brain):
    COMB_ID = 0

    def __init__(self, name, ninputs, nactions, brainA, brainB,
                 prob=0.5, directory="save", rewardbuffer=None):
        super().__init__(name, ninputs, nactions,
                         directory=directory, rewardbuffer=rewardbuffer)
        self.brainA, self.brainB = brainA, brainB
        self.p = prob

    def think(self, inputs, memory):
        if random.random() < self.p:
            return self.brainA.think(inputs, memory)
        else:
            return self.brainB.think(inputs, memory)

    def train(self, iters, batch, totreward=None):
        self.brainA.train(iters=iters, batch=batch, totreward=totreward)
        self.brainB.train(iters=iters, batch=batch, totreward=totreward)

    def startup(self):
        super().startup()
        self.brainA.startup()
        self.brainB.startup()

    def cleanup(self):
        super().cleanup()
        self.brainA.cleanup()
        self.brainB.cleanup()

    @staticmethod
    def make_combined_constructor(brainconsA, brainconsB, p=0.5):
        """
        Makes a constructor to combine the two brain constructors given
        :param brainconsA: constructor for brain A
        :param brainconsB: constructor for brain B
        :param p: p of choosing A over B
        :return: constructor for A-B hybrid
        """
        def constructor(name, ninputs, nactions,
                        directory='save', rewardbuffer=None):
            if rewardbuffer is None:
                rewardbuffer = RewardBuffer("CB{}_{}".format(CombinedBrain.COMB_ID,name),ninputs)
            brainA = brainconsA("C{}A_{}".format(CombinedBrain.COMB_ID,name),
                                ninputs, nactions,
                                directory=directory, rewardbuffer=rewardbuffer)
            brainB = brainconsB("C{}B_{}".format(CombinedBrain.COMB_ID,name),
                                ninputs, nactions,
                                directory=directory, rewardbuffer=rewardbuffer)
            return CombinedBrain(name, ninputs, nactions, brainA, brainB, p,
                                 directory=directory, rewardbuffer=rewardbuffer)
        return constructor

    def print_diag(self, sample_in):
        self.brainA.print_diag(sample_in)
        self.brainB.print_diag(sample_in)

    def debug(self, debuginput,debugmemory):
        deba = self.brainA.debug(debuginput, )
        if deba is not None:
            return deba
        return self.brainB.debug(debuginput, )
