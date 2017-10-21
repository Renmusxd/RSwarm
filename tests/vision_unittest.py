import unittest
import numpy

from world import World
from brain import ToyBrain
from bot import Bot

class VisionTestSuite(unittest.TestCase):

    def test_center_dist(self):
        POSX, POSY = 100,100
        world = World(ToyBrain, ToyBrain)

        for i in range(360):
            b = world.make_prey(POSX,POSY,i)
            x,y = get_pos_d_from(POSX,POSY,i,Bot.VIEW_DIST/2.0)
            c = world.make_prey(x,y,0)

            _,_,distances = Bot.split_senses(b.senses())
            center_dist = distances[int(len(distances)/2)]

            self.assertGreater(center_dist, 0.49)
            self.assertLess(center_dist, 0.51)

            world.reset()

    def test_side_dist(self):
        POSX, POSY = 100, 100
        world = World(ToyBrain, ToyBrain)

        for i in range(360):
            b = world.make_prey(POSX, POSY, i)
            x1, y1 = get_pos_d_from(POSX, POSY, i + 1 - Bot.FOV, Bot.VIEW_DIST / 2.0)
            x2, y2 = get_pos_d_from(POSX, POSY, i - 1 + Bot.FOV, Bot.VIEW_DIST / 4.0)
            c1 = world.make_prey(x1, y1, 0)
            c2 = world.make_prey(x2, y2, 0)

            _, _, distances = Bot.split_senses(b.senses())
            left_dist = distances[0]
            right_dist = distances[-1]

            self.assertGreater(left_dist, 0.49)
            self.assertLess(left_dist, 0.51)

            self.assertGreater(right_dist, 0.24)
            self.assertLess(right_dist, 0.26)

            world.reset()

    def test_map_leftedge(self):
        world = World(ToyBrain, ToyBrain)

        # Face towards wall
        b = world.make_prey(Bot.VIEW_DIST/2.0, Bot.VIEW_DIST, 180)
        _, _, distances = Bot.split_senses(b.senses())
        center_dist = distances[int(len(distances) / 2)]

        self.assertEqual(center_dist,0.5)

    def test_map_rightedge(self):
        world = World(ToyBrain, ToyBrain)

        # Face towards wall
        b = world.make_prey(world.width() - Bot.VIEW_DIST/2.0, Bot.VIEW_DIST, 0)
        _, _, distances = Bot.split_senses(b.senses())
        center_dist = distances[int(len(distances) / 2)]

        self.assertEqual(center_dist, 0.5)

    def test_map_topedge(self):
        world = World(ToyBrain, ToyBrain)

        # Face towards wall
        b = world.make_prey(Bot.VIEW_DIST, world.height() - Bot.VIEW_DIST/2., 90)
        _, _, distances = Bot.split_senses(b.senses())
        center_dist = distances[int(len(distances) / 2)]

        self.assertEqual(center_dist,0.5)

    def test_map_botedge(self):
        world = World(ToyBrain, ToyBrain)

        # Face towards wall
        b = world.make_prey(Bot.VIEW_DIST, Bot.VIEW_DIST/2.0, 270)
        _, _, distances = Bot.split_senses(b.senses())
        center_dist = distances[int(len(distances) / 2)]

        self.assertEqual(center_dist,0.5)


def get_pos_d_from(x, y, d, dist):
    dx = numpy.cos(numpy.deg2rad(d)) * dist
    dy = numpy.sin(numpy.deg2rad(d)) * dist
    return x + dx, y + dy


if __name__ == '__main__':
    unittest.main()
