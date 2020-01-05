from env.ChainReaction import Game
import numpy as np
def transform(game: Game):
    """

    :param game: game state to be transformed
    :return: one hot image for all players where game.curPlayer is treated as the first player
    """
    state = np.zeros((game.totalRows, game.totalCols, game.numPlayers))
    offset = game.curPlayer
    for i in range(game.totalRows):
        for j in range(game.totalCols):
            if(not(game.grid[i][j].isEmpty)):
                state[i][j][(game.grid[i][j].playerIdx + offset) % game.numPlayers] = game.grid[i][j].numOrbs

    return state