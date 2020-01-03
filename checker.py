from env.ChainReaction import Cell, Game

env = Game(3,3)

env.printGrid()
env._putOrb(0, 2, 2)
env._putOrb(0, 2, 2)
env._putOrb(0, 1, 2)
env._putOrb(0, 2, 1)
env._putOrb(1, 1, 1)
env._putOrb(1, 1, 1)
env._putOrb(1, 1, 1)
env.printGrid()
env._putOrb(1, 1, 1)
env.printGrid()