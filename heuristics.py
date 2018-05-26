from brain import Brain
from bot import Bot
import sys


class PredHeuristicBrain(Brain):

    def __init__(self, name, ninputs, nactions, **kwargs):
        super().__init__(name, ninputs, nactions, **kwargs)

    def think(self, inputs, memory):
        return {bid: Bot.make_actions_from_label(self.think_indiv(inputs[bid]))
                for bid in inputs}, {}

    def think_indiv(self,inputs):
        inputs, vision, distance = Bot.split_senses(inputs)
        linputs = Bot.label_inputs(inputs)
        predgroups, preygroups = group_bot_vision(vision, distance)
        preddir, preydir = closest_bot_directions(predgroups, preygroups)

        directions = ['lmov','forward','rmov']
        if preydir == -1:
            # Wander
            pass
        else:
            return directions[preydir]

        # Order indexes by distance
        return 'still'

    def train(self, iters, batch, **kwargs):
        pass


class PreyHeuristicBrain(Brain):

    def __init__(self, name, ninputs, nactions, **kwargs):
        super().__init__(name, ninputs, nactions, **kwargs)

    def think(self, inputs, memory):
        return {bid: Bot.make_actions_from_label(self.think_indiv(inputs[bid]))
                for bid in inputs}, {}

    def think_indiv(self, inputs):
        inputs, vision, distance = Bot.split_senses(inputs)
        linputs = Bot.label_inputs(inputs)
        predgroups, preygroups = group_bot_vision(vision, distance)

        # Priorities
        if linputs['energy'] < 0.5:
            return 'eat'
        if linputs['tile'] < 0.75:
            return 'forward'
        if sum(vision[0]) == 0 and distance[0] < 1.0:
            return 'rmov'
        if sum(vision[-1]) == 0 and distance[-1] < 1.0:
            return 'lmov'
        if sum(vision[-2]) != 0 and distance[-2] < 1.0:
            return 'lmov'
        if sum(vision[1]) != 0 and distance[1] < 1.0:
            return 'rmov'
        # Other things
        return 'forward'

    def train(self, iters, batch, **kwargs):
        pass


def group_bot_vision(vis,dist,mode='gray'):
    """
    Group other seen bots into left,center,right
    :param vis:
    :param dist:
    :param mode:
    :return: (leftpreds,centerpreds,rightpreds) (...preys...) arrays of distance sorted
    """
    if mode == 'gray':
        PREDVAL = -1
        PREYVAL = 1

        # Left, Center, Right
        preds = [[], [], []]
        preys = [[], [], []]

        middleindx = int(len(dist) / 2.0)
        for i, (visval, distval) in enumerate(zip(vis,dist)):
            if visval not in [PREDVAL, PREYVAL]:
                continue
            # Choose appropriate array
            botarr = preds if visval == PREDVAL else preys
            # Left
            if i < middleindx:
                indx = 2
            # Center
            elif i == middleindx:
                indx = 1
            # Right
            else:
                indx = 0
            # Put in container
            botarr[indx].append(distval)
        for i in range(len(preds)):
            preds[i] = list(sorted(preds[i]))
            preys[i] = list(sorted(preys[i]))

        return tuple(preds), tuple(preys)


def closest_bot_directions(predtuples, preytuples):
    """
    Returns closest of each type of bot
    :param predtuples: output from group_bot_vision
    :param preytuples: output from group_bot_vision
    :return: -1, 0, 1, 2 for each -> None, Left, Center, Right
    """
    outputs = []
    for directions in [predtuples, preytuples]:
        if sum(map(lambda s: len(s), directions)) == 0:
            outputs.append(-1)
        else:
            closest_dir = -1
            closest_dist = sys.maxsize
            for i,direction in enumerate(directions):
                if len(direction) > 0:
                    if direction[0] < closest_dist:
                        closest_dist = direction[0]
                        closest_dir = i
            outputs.append(closest_dir)
    return tuple(outputs)
