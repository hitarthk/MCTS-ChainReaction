from texttable import Texttable
import copy
from configs.defaultConfigs import config
class Cell(object):
    def __init__(self, row, col, totalRows = 5, totalCols = 5):
        self.isEmpty = True
        self.playerIdx = -1
        self.numOrbs = 0
        self.row = row
        self.col = col
        self.maxCapacity = self.__getCapacity(row, col, totalRows, totalCols)

    def __getCapacity(self, row, col, totalRows, totalCols):
        if((row==0 or row==totalRows-1) and (col==0 or col==totalCols-1)):
            return 2
        elif(row==0 or row==totalRows-1 or col==0 or col==totalCols-1):
            return 3
        else:
            return 4

    def putOrb(self, playerIdx):
        self.isEmpty = False
        self.numOrbs += 1
        self.playerIdx = playerIdx

    def reset(self):
        self.isEmpty = True
        self.numOrbs = 0
        self.playerIdx = -1

    def getString(self):
        if(self.isEmpty):
            return "   "
        else:
            return str(self.playerIdx)+":"+str(self.numOrbs)

class Game(object):
    def __init__(self, totalRows = 5, totalCols = 5, numPlayers = 2):
        self.totalRows = totalRows
        self.totalCols = totalCols
        self.numPlayers = numPlayers
        self.grid = [[Cell(i, j, self.totalRows, self.totalCols) for j in range(self.totalCols)] for i in range(self.totalRows)]
        self.di = [-1, 0, 1, 0]
        self.dj = [0, -1, 0, 1]
        self.curPlayer = 0
        self.totalMoves = 0
        self.validMoves = []
        self.validMovesForMove = -1

    def makeMove(self, row, col):
        if(not(self.grid[row][col].isEmpty) and self.grid[row][col].playerIdx != self.curPlayer):
            raise PermissionError(f'Invalid move. curPlayer: {self.curPlayer} | occupied by: {self.grid[row][col].playerIdx}')
        self._putOrb(self.curPlayer, row, col)
        self.totalMoves += 1
        self.curPlayer = self.totalMoves % self.numPlayers

    def _putOrb(self, playerIdx, row, col):
        curCell = self.grid[row][col]
        curCell.putOrb(playerIdx)
        self._explode(curCell, playerIdx)

    def _explode(self, cell, playerIdx):
        if(cell.numOrbs==cell.maxCapacity):
            i = cell.row
            j = cell.col
            di = self.di
            dj = self.dj
            totalRows = self.totalRows
            totalCols = self.totalCols
            cell.reset()
            for k in range(4):
                ni = i + di[k]
                nj = j + dj[k]
                if(ni>=0 and ni<totalRows and nj>=0 and nj<totalCols):
                    self._putOrb(playerIdx, ni, nj)

    def printGrid(self):
        table = Texttable()
        sg = [[self.grid[i][j].getString() for j in range(self.totalCols)] for i in range(self.totalRows)]
        table.add_rows(sg)
        print(table.draw())
        print("||||||||||||||||||||||||||||||||||")
        print()
        print()

    def getReward(self):
        cnt = 0
        for i in range(self.totalRows):
            for j in range(self.totalCols):
                curCell = self.grid[i][j]
                if curCell.isEmpty:
                    return (0, False)
                cnt += curCell.numOrbs if curCell.playerIdx==self.curPlayer else 0

        if(cnt==0 and self.totalMoves>self.curPlayer):
            return (-1, True)
        else:
            return (0, False)

    def getValidMoves(self):
        if(self.totalMoves != self.validMovesForMove):
            self.validMoves = []
            for i in range(self.totalRows):
                for j in range(self.totalCols):
                    if(self.grid[i][j].isEmpty or self.grid[i][j].playerIdx==self.curPlayer):
                        self.validMoves.append((i, j))
        return self.validMoves

    def getNextState(self, move = None, row = -1, col = -1):
        assert move!=None or (row>=0 and col>=0)
        if(move is None):
            move = (row, col)
        newGame = copy.deepcopy(self)
        newGame.makeMove(move[0], move[1])
        return newGame


def getNewGame():
    return Game(config.totalRows, config.totalCols, config.numPlayers)