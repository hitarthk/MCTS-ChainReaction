from texttable import Texttable

class Cell(object):
    def __init__(self, row, col, totalRows = 9, totalCols = 6):
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
    def __init__(self, totalRows = 9, totalCols = 6, numPlayers = 2):
        self.totalRows = totalRows
        self.totalCols = totalCols
        self.numPlayers = numPlayers
        self.grid = [[Cell(i, j, self.totalRows, self.totalCols) for j in range(self.totalCols)] for i in range(self.totalRows)]
        self.di = [-1, 0, 1, 0]
        self.dj = [0, -1, 0, 1]
        self.curPlayer = 0
        self.totalMoves = 0

    def makeMove(self, row, col):
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