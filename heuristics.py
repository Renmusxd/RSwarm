from brain import Brain
from bot import Bot

class PredHeuristicBrain(Brain):

    def __init__(self, name, ninputs, nactions):
        super().__init__(name, ninputs, nactions)

    def think(self, inputs):
        inputs, vision, distance = Bot.split_senses(inputs)
        linputs = Bot.label_inputs(inputs)
        # TODO: predatory actions

    def train(self, iters=1000, batch=64):
        pass


class PreyHeuristicBrain(Brain):

    def __init__(self, name, ninputs, nactions):
        super().__init__(name, ninputs, nactions)

    def think(self, inputs):
        inputs, vision, distance = Bot.split_senses(inputs)
        linputs = Bot.label_inputs(inputs)
        # TODO: prey actions

    def train(self, iters=1000, batch=64):
        pass
