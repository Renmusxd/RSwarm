import unittest
from tfbrain import RewardBuffer
import numpy


class TestRewardBuffer(unittest.TestCase):

    BATCHSIZE = 100
    BUFFERSIZE = 1000

    STATESIZE = 3
    NACTIONS = 5

    def testfillbatch(self):
        # Buffersize > batch
        buff = RewardBuffer(buffersize=2*TestRewardBuffer.BATCHSIZE+1, removebatch=TestRewardBuffer.BATCHSIZE)

        inputdict, actiondict, rewarddict, newstatedict = makebufferfiller(TestRewardBuffer.BATCHSIZE,
                                                                           TestRewardBuffer.STATESIZE,
                                                                           TestRewardBuffer.NACTIONS)

        buff.reward(inputdict,actiondict,rewarddict,newstatedict)
        self.assertEqual(len(buff),TestRewardBuffer.BATCHSIZE)

    def testmultiiter(self):
        # Buffersize > 2*batch
        buff = RewardBuffer(buffersize=2*TestRewardBuffer.BATCHSIZE+1, removebatch=TestRewardBuffer.BATCHSIZE)

        inputdict, actiondict, rewarddict, newstatedict = makebufferfiller(TestRewardBuffer.BATCHSIZE,
                                                                           TestRewardBuffer.STATESIZE,
                                                                           TestRewardBuffer.NACTIONS)

        buff.reward(inputdict, actiondict, rewarddict, newstatedict)
        self.assertEqual(len(buff), TestRewardBuffer.BATCHSIZE)

        inputdict, actiondict, rewarddict, newstatedict = makebufferfiller(TestRewardBuffer.BATCHSIZE,
                                                                           TestRewardBuffer.STATESIZE,
                                                                           TestRewardBuffer.NACTIONS)

        buff.reward(inputdict, actiondict, rewarddict, newstatedict)
        self.assertEqual(len(buff), 2*TestRewardBuffer.BATCHSIZE)

    def testfill(self):
        buff = RewardBuffer(buffersize=2 * TestRewardBuffer.BATCHSIZE + 1, removebatch=TestRewardBuffer.BATCHSIZE)

        expectedsize = 0

        while expectedsize < 2*TestRewardBuffer.BUFFERSIZE:
            inputdict, actiondict, rewarddict, newstatedict = makebufferfiller(TestRewardBuffer.BATCHSIZE,
                                                                               TestRewardBuffer.STATESIZE,
                                                                               TestRewardBuffer.NACTIONS)

            buff.reward(inputdict, actiondict, rewarddict, newstatedict)
            expectedsize = expectedsize + TestRewardBuffer.BATCHSIZE
            self.assertLessEqual(len(buff), TestRewardBuffer.BUFFERSIZE)

    def testgen(self):
        # Buffersize > batch
        buff = RewardBuffer(buffersize=2 * TestRewardBuffer.BATCHSIZE + 1, removebatch=TestRewardBuffer.BATCHSIZE)

        inputdict, actiondict, rewarddict, newstatedict = makebufferfiller(TestRewardBuffer.BATCHSIZE,
                                                                           TestRewardBuffer.STATESIZE,
                                                                           TestRewardBuffer.NACTIONS)
        flatinputs = [tuple(e_in) for e_in in inputdict.values()]
        flatactions = [e_act for e_act in actiondict.values()]
        flatrewards = [e_rew for e_rew in rewarddict.values()]
        flatnews = [tuple(e_new) for e_new in newstatedict.values()]

        buff.reward(inputdict, actiondict, rewarddict, newstatedict)

        niters_target = 100
        niters = 0
        for ins, acts, rews, news in buff.get_batch_gen(TestRewardBuffer.BUFFERSIZE,niters_target):
            for e_in, e_act, e_rew, e_new in zip(ins, acts, rews, news):
                self.assertTrue(tuple(e_in) in flatinputs)
                self.assertTrue(e_act in flatactions)
                self.assertTrue(e_rew in flatrewards)
                self.assertTrue(tuple(e_new) in flatnews)
            niters += 1
        self.assertEqual(niters, niters_target)


def makebufferfiller(n, nstate, naction):
    eids = numpy.arange(0, n)
    inputs = numpy.random.uniform(0, 1, (n, nstate))
    actions = numpy.random.randint(0, naction, n)
    rewards = numpy.random.uniform(0, 100, n)
    newstates = numpy.random.uniform(0, 1, (n, nstate))

    inputdict = {}
    actiondict = {}
    rewarddict = {}
    newstatedict = {}
    for i, eid in enumerate(eids):
        inputdict[eid] = inputs[i, :]
        actiondict[eid] = actions[i]
        rewarddict[eid] = rewards[i]
        newstatedict[eid] = newstates[i, :]
    return inputdict, actiondict, rewarddict, newstatedict

if __name__ == '__main__':
    unittest.main()