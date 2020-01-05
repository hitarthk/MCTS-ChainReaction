from algo.MCTS import *
class Buffer(object):
    def __init__(self, totalRows = 5, totalCols = 5, numPlayers = 2):
        self.totalRows = totalRows
        self.totalCols = totalCols
        self.numPlayers = numPlayers
        self.states = np.empty((0, totalRows, totalCols, numPlayers), dtype=float)
        self.actions = np.empty((0, 2), dtype=int)
        self.playerIdx = np.empty((0,), dtype=int)
        self.intuitionProbs = np.empty((0, totalRows * totalCols), dtype=float)
        self.analysisProbs = np.empty((0, totalRows * totalCols), dtype=float)
        self.rewards = np.empty((0), dtype=float)

    def addData(self, gameTrace: list(TreeNode)):
        for i in range(len(gameTrace)):
            treeNode = gameTrace[i]
            state = self.transform(treeNode.game)
            self.states = np.append(self.states, np.expand_dims(state, axis=0), axis = 0)
            self.actions = np.append(self.actions, np.expand_dims(
                                     np.array([treeNode.selectedAction.row, treeNode.selectedAction.col]), axis = 0), axis = 0)
            self.playerIdx = np.append(self.playerIdx, treeNode.selectedAction.playerIdx)
            self.intuitionProbs = np.append(self.intuitionProbs, np.expand_dims(treeNode.intuitionProbs, axis = 0), axis = 0)
            self.analysisProbs = np.append(self.analysisProbs, np.expand_dims(treeNode.analysisProbs, axis = 0), axis = 0)
            self.rewards = np.append(self.rewards, treeNode.outcome)

def transform(game: Game):
    """

    :param game: game state to be transformed
    :return: one hot image for all players where game.curPlayer is treated as the first player
    """
    state = np.zeros((game.numPlayers, game.totalRows, game.totalCols))
    offset = game.curPlayer
    for i in range(game.totalRows):
        for j in range(game.totalCols):
            if(not(game.grid[i][j].isEmpty)):
                state[i][j][(game.grid[i][j].playerIdx + offset) % game.numPlayers] = game.grid[i][j].numOrbs

    return state