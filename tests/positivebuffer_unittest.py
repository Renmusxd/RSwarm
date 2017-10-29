import unittest
import numpy

from world import World
from brain import Brain
from bot import Bot


class QueueBrain(Brain):
    def __init__(self, name, ninputs, nactions):
        super().__init__(name, ninputs, nactions)
        self.actlam = lambda _: 'still'
        self.repeat = True

    def think(self, inputs):
        if self.actlam is None:
            raise Exception("No action specified and repeat set to false")
        acts = {k: Bot.make_actions_from_label(self.actlam(k)) for k in inputs}
        if not self.repeat:
            self.actlam = None
        return acts

    def queue_action(self, actlam, repeat=False):
        self.actlam = actlam
        self.repeat = repeat

    def train(self, iters, batch, totreward=None):
        pass


class AttackTestSuite(unittest.TestCase):

    def test_center_dist(self):
        POSX, POSY = 100,100

        world = World(QueueBrain, QueueBrain, restockbots=False)
        predbrain = world.predbrain
        preybrain = world.preybrain

        for i in range(360):
            b = world.make_pred(POSX,POSY,i)
            x,y = get_pos_d_from(POSX,POSY,i,Bot.ACTION_RADIUS/2.0)
            c = world.make_prey(x,y,0)

            predbrain.queue_action(lambda _: 'atck')
            world.update(0, b.id)

            self.assertEqual(world.stats['Attacks'], 1)

            world.reset()


def get_pos_d_from(x, y, d, dist):
    dx = numpy.cos(numpy.deg2rad(d)) * dist
    dy = numpy.sin(numpy.deg2rad(d)) * dist
    return x + dx, y + dy


if __name__ == '__main__':
    unittest.main()
