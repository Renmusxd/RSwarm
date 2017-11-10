import unittest
import numpy

import random
from rewardbuffer import PositiveReserveBuffer


class PositiveTestSuite(unittest.TestCase):

    def test_random_ensure(self):
        N = 1000
        R = 0.5
        MAXP = int(R*N)
        MAXN = int((1-R)*N)

        r = PositiveReserveBuffer('test', 1, maxposratio=0.5, buffersize=int(N/2))

        lastnegatives = 0
        lastpositives = 0
        for i in range(N):
            toadd = 1 if random.randint(0,1) == 0 else -1

            rew(r, toadd)
            uniques, counts = numpy.unique(r.rewards[:i+1], return_counts=True)

            # Make sure only -1 and 1
            self.assertLessEqual(len(uniques),2, msg="Only 1/-1 allowed in {}".format(uniques))

            newpositives = 0
            newnegatives = 0
            for item, count in zip(uniques, counts):
                if item < 0:
                    newnegatives = count
                elif item > 0:
                    newpositives = count

            if toadd > 0:
                self.assertTrue(lastpositives == newpositives or lastpositives == newpositives - 1,
                                "Positive didn't increase or maintain count: {}-{}".format(lastpositives,newpositives))
                # Positive may evict negative, otherwise would include
                # self.assertTrue(lastnegatives == newnegatives,
                #                 "Positive changed negative count")
            elif toadd < 0:
                self.assertTrue(lastnegatives == newnegatives or lastnegatives == newnegatives - 1,
                                "Negative didn't increase or maintain count: {}-{}".format(lastnegatives,newnegatives))
                self.assertTrue(lastpositives == newpositives,
                                "Negative changed positive count: {}-{}".format(lastpositives,newpositives))

            lastnegatives, lastpositives = newnegatives, newpositives


def rew(r,x):
    r.reward({0: 0}, {0: 0}, {0: x}, {0: 0})


if __name__ == '__main__':
    unittest.main()
