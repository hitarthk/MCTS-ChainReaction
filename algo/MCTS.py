import numpy as np
import time
import gc
from env.ChainReaction import Cell, Game, getNewGame
from configs.defaultConfigs import config
from models.ResnetFeatures import IntuitionPolicy

#Use this class to represent any generic action. Not using this now to speed up implementation
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
    def __init__(self, nextNode: TreeNode, row: int, col: int, p: float, pi: float):
        self.nextNode = nextNode
        self.action = (row, col)
        self.n = 0 # visit count
        self.w = 0 # total action value
        self.q = 0 # average action value
        self.p = p # prior probability assigned by the Intuition policy
        self.pi = pi

    def addObservation(self, estimatedValue):
        self.w += estimatedValue
        self.n += 1
        self.q = self.w/self.n

class SearchTree(object):
    def __init__(self, root: TreeNode, intuitionPolicy: IntuitionPolicy):
        self.gameTrace = list()
        self.root = root
        self.intuitionPolicy = intuitionPolicy
        self.cUB = config.cUB
        self.simulationsPerMove = config.simulationsPerMove
        self.temperature = config.initialTemperature

    def _flattenMove(self, move):
        return move[0]*self.root.game.totalCols + move[1]

    def _unflattenMove(self, flattenedMove):
        return (flattenedMove//self.root.game.totalCols, flattenedMove%self.root.game.totalCols)

    def _getFlattenedValidMovesMask(self, validMoves):
        mask = np.zeros(shape=(self.root.game.totalRows, self.root.game.totalCols), dtype = float)
        validMoves = tuple(zip(*validMoves))
        mask[validMoves] = 1.0
        return mask.flatten()

    def sanitizeActionProbs(self, actionProbs, validMoves):
        validMovesMask = self._getFlattenedValidMovesMask(validMoves)
        actionProbs = actionProbs * validMovesMask
        totalProbs = np.sum(actionProbs)
        actionProbs /= totalProbs
        return actionProbs

    #sets the selectedAction, analysisProbs and intuitionProbs in the root node. returns nothing
    def searchMove(self):
        for i in range(self.simulationsPerMove):
            self.select(self.root)

        analysisProbs = np.zeros(shape=(self.root.game.totalRows * self.root.game.totalCols), dtype=float)
        for move, edge in self.root.edges.items():
            analysisProbs[self._flattenMove(move)] = edge.n**(1/self.temperature)

        totalProb = analysisProbs.sum()
        analysisProbs /= totalProb

        self.root.analysisProbs = analysisProbs
        sampledMove = np.random.choice(self.root.game.totalRows*self.root.game.totalCols, 1, p = analysisProbs)[0]
        self.root.selectedAction = self._unflattenMove(sampledMove)


    def select(self, curNode: TreeNode):
        """

        :param curNode:
        """
        intuitionProbs = curNode.intuitionProbs
        validMoves = curNode.game.getValidMoves()

        #if curNode is a leaf node then call expand on it
        if(len(curNode.edges)==0):
            return self.expand(curNode)


        bestMove = None
        bestUCB = -1

        totalN = 0
        for move, edge in curNode.edges.items():
            totalN += edge.n

        for move, edge in curNode.edges.items():
            ucb = edge.q + self.cUB*edge.p*(np.sqrt(totalN+1)/(edge.n+1))
            if(ucb>=bestUCB):
                bestUCB = ucb
                bestMove = move

        # multiply -1 because every select method returns the state evaluation for the node itself.
        # The current node is adversary of its child. Hence less for child is more for parent (how ironic :P)
        estimatedValue = -1 * self.select(curNode.edges[bestMove].nextNode)
        curNode.edges[bestMove].addObservation(estimatedValue)
        return estimatedValue

    def expand(self, curNode: TreeNode):
        if(curNode.game.getReward()[1]==True):
            return -1
        policyInput = transform(curNode.game)
        policyInput = np.expand_dims(policyInput, axis=0)
        (intuitionProbs, intuitionValue, _) = self.intuitionPolicy(policyInput)
        intuitionProbs = intuitionProbs.numpy()
        intuitionProbs = np.squeeze(intuitionProbs)
        validMoves = curNode.game.getValidMoves()
        intuitionProbs = self.sanitizeActionProbs(intuitionProbs, validMoves)
        curNode.intuitionProbs = intuitionProbs

        for move in validMoves:
            nextState = curNode.game.getNextState(move)
            nextNode = TreeNode(nextState)
            curNode.edges[move] = TreeEdge(nextNode, move[0], move[1], intuitionProbs[self._flattenMove(move)], 0.)

        return intuitionValue.numpy()

    def next(self):
        """
        Moves the actual game ahead by 1 step. A move is searched by mcts and stored in the current root
        :return: (reward, isTerminal): a tuple indicating the reward and the fact if the state is terminal
        """
        self.root.game.printGrid()
        t1 = time.time()
        self.searchMove()
        t2 = time.time()
        print(f'Played move: {self.root.selectedAction}. Time taken: {t2-t1}')
        self.gameTrace.append(self.root)
        newRoot = self.root.edges[self.root.selectedAction].nextNode
        self.root.edges = {}
        self.root = newRoot
        gc.collect()
        return self.root.game.getReward()

from utils.buffer import *
from utils.misc import transform

class ExperienceCollector(object):
    def __init__(self):
        self.buffer = Buffer(config.totalRows, config.totalCols, config.numPlayers)

    def collectExperience(self, numGames: int, intuitionPolicy: IntuitionPolicy):
        """
        Call to collect experience. Right now only experience collection with 2 players is supported. Do not call this with numPlayers>2
        :param numGames: number of self-play games before policy evaluation and improvement stage
        """
        for iter in range(numGames):
            searchTree = SearchTree(TreeNode(getNewGame()), intuitionPolicy)
            rewardTuple = (0, False)
            while(rewardTuple[1] == False):
                rewardTuple = searchTree.next()

            gameTrace = searchTree.gameTrace
            multiplier = 1
            for i in range(len(gameTrace)-1, -1, -1):
                gameTrace[i].outcome = -1*multiplier
                multiplier *= -1

            self.buffer.addData(searchTree.gameTrace)

if __name__=='__main__':
    exp = ExperienceCollector()
    exp.collectExperience(2, IntuitionPolicy(config.totalRows, config.totalCols, config.numPlayers,
                                             10, 64))