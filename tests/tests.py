from world import World
from brain import ToyBrain
from heuristics import *

def main():
    model = World(PredHeuristicBrain, PreyHeuristicBrain)
    a = model.make_pred(100,100, 90)
    b = model.make_prey(50,101,0)

    print(model.predbrain.think_indiv(a.senses()))

if __name__ == "__main__":
    main()
