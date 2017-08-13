from abc import ABCMeta, abstractmethod
import random
import os


class Brain(metaclass=ABCMeta):

    def __init__(self, name, ninputs, nactions, directory='save'):
        """
        Construct a new brain
        :param ninputs: number of inputs
        :param nactions: number of possible actions
        """
        self.ninputs = ninputs
        self.nactions = nactions
        self.name = name
        self.directory = directory

    def __init_subclass__(cls, **kwargs):
        pass

    @abstractmethod
    def save(self):
        """
        Saves parameters to default file.
        """
        raise NotImplementedError

    @abstractmethod
    def think(self, inputs):
        """
        Provides actions for inputs
        :param inputs: dictionary of id:input to think about
        :return: dictionary of id:actions
        """
        raise NotImplementedError

    @abstractmethod
    def reward(self, inputs, actions, rewards, newinputs):
        """
        Rewards last actions using Q learning approach
        :param inputs:
        :param actions:
        :param rewards:
        :param newinputs:
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, iters=1000, batch=64):
        """
        Train the brain for a bit based in rewards previously provided
        :param iters:
        :param batch:
        :return:
        """
        raise NotImplementedError


    @abstractmethod
    def startup(self):
        raise NotImplementedError

    @abstractmethod
    def cleanup(self):
        raise NotImplementedError

    @abstractmethod
    def print_diag(self, sample_in):
        raise NotImplementedError


class ToyBrain(Brain):

    def __init__(self, ninputs, nactions, directory):
        super().__init__('toy', ninputs, nactions, directory)

    def save(self):
        pass

    def think(self, inputs):
        return {entityid: random.randint(0, self.nactions-1)
                for entityid in inputs.keys()}

    def train(self, iters=1000, batch=64):
        pass

    def reward(self, inputs, actions, rewards, newinputs):
        pass

    def startup(self):
        pass

    def cleanup(self):
        pass

    def print_diag(self, sample_in):
        pass