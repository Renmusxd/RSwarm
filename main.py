from world import World
from brain import ToyBrain, CombinedBrain
from tfbrain import TFBrain


def make_brain_constructor(predprey):
    """

    :param predprey: string "pred" or string "prey"
    :return:
    """
    constructor = CombinedBrain.make_combined_constructor(TFBrain,ToyBrain,0.9)
    return constructor

def make_model():
    world = World(make_brain_constructor('pred'), make_brain_constructor('prey'))
    return world


if __name__ == "__main__":
    model = make_model()
    try:
        model.startup()
        while True:
            model.update(1)
    finally:
        print("Cleaning up...")
        model.cleanup()
