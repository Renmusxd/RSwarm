from world import World
from tfbrain import TFBrain


def make_model():
    world = World(TFBrain)
    world.make_bot(50, 50)
    world.make_bot(100, 50)
    return world

if __name__ == "__main__":

    model = make_model()
    while True:
        model.update(1)
