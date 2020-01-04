import numpy as np
import gc
from env.ChainReaction import Cell, Game, getNewGame
from utils.buffer import transform, Buffer
from configs.defaultConfigs import config
from models.ResnetFeatures import IntuitionPolicy

class Action(object):
    def __init__(self, playerIdx: int, row: int, col: int):
        self.playerIdx = playerIdx
        self.row = row
        self.col = col

    def __hash__(self):
        return hash((self.playerIdx, self.row, self.col))

    def __eq__(self, other):
        return (self.playerIdx, self.row, self.col) == (other.playerIdx, other.row, other.col)

    def __ne__(self, other):
        return not(self==other)


class TreeNode(object):
    def __init__(self, game: Game):
        self.edges = {}
        self.game = game
        self.outcome = 0
        self.selectedAction = None
        self.intuitionProbs = None
        self.analysisProbs = None

class TreeEdge(object):
    def __init__(self, nextNode: TreeNode, playerIdx: int, row: int, col: int, p: float, pi: float):
        self.nextNode = nextNode
        self.action = Action(playerIdx, row, col)
        self.n = 0 # visit count
        self.w = 0 # total action value
        self.q = 0 # average action value
        self.p = p # prior probability assigned by the Intuition policy
        self.pi = pi

class SearchTree(object):
    def __init__(self, root: TreeNode, intuitionPolicy: IntuitionPolicy):
        self.gameTrace = list(TreeNode)
        self.root = root
        self.intuitionPolicy = intuitionPolicy


    #sets the selectedAction, analysisProbs and intuitionProbs in the root node. returns nothing
    def searchMove(self):
        intuitionProbs = self.root.intuitionProbs

        pass

    def select(self, curNode: TreeNode):
        """

        :param curNode:
        """
        pass

    def expand(self, curNode: TreeNode):
        policyInput = transform(curNode.game)
        policyInput = np.expand_dims(policyInput, axis=0)
        intuitionProbs = self.intuitionPolicy(policyInput)
        pass


    def next(self):
        """
        Moves the actual game ahead by 1 step. A move is searched by mcts and stored in the current root
        :return: (reward, isTerminal): a tuple indicating the reward and the fact if the state is terminal
        """
        self.searchMove()
        self.gameTrace.append(self.root)
        newRoot = self.root.edges[self.root.selectedAction].nextNode
        self.root.edges = {}
        self.root = newRoot
        gc.collect()
        return self.root.game.getReward()

class ExperienceCollector(object):
    def __init__(self):
        self.buffer = Buffer(config.totalRows, config.totalCols, config.numPlayers)

    def collectExperience(self, numGames):
        """
        Call to collect experience. Right now only experience collection with 2 players is supported. Do not call this with numPlayers>2
        :param numGames: number of self-play games before policy evaluation and improvement stage
        """
        for iter in range(numGames):
            searchTree = SearchTree(TreeNode(getNewGame()))
            rewardTuple = (0, False)
            while(rewardTuple[1] == False):
                rewardTuple = searchTree.next()

            gameTrace = searchTree.gameTrace
            multiplier = 1
            for i in range(len(gameTrace)-1, -1, -1):
                gameTrace[i].outcome = -1*multiplier
                multiplier *= -1

            self.buffer.addData(searchTree.gameTrace)


