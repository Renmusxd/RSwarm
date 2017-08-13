from world import World
from tfbrain import TFBrain


def make_model():
    world = World(TFBrain,TFBrain)
    return world

if __name__ == "__main__":

    model = make_model()
    model.startup()
    while True:
        model.update(1)
