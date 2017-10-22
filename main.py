from world import *

running = True


if __name__ == "__main__":
    model = make_model()
    try:
        model.startup()
        while True:
            model.update(1)
    finally:
        print("Cleaning up...")
        model.cleanup()
