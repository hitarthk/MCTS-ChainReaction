from env.ChainReaction import Cell, Game
import numpy as np
import sys
env = Game(3,3)
env.makeMove(0, 0)
nextEnv = env.getNextState(move = (1, 1))
env.makeMove(2, 2)
nextEnv.makeMove(2, 1)
env.printGrid()

nextEnv.printGrid()
mask = np.zeros((3,3))
idx = [(0,0), (1, 1), (2, 2)]
idx2 = tuple(zip(*idx))
print(idx2)
mask[idx2] = 1
print(mask)

print(sys.getrecursionlimit())