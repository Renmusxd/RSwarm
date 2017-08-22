from world import World
from tfbrain import TFBrain


def make_model():
    world = World(TFBrain,TFBrain)
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