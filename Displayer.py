from BaseDisplayer import BaseDisplayer
import platform
import os

colorMap = { 0: 100,
             1: 102,
             2: 105,
             -1: 40}

# cTemp = "\x1b[%dm%7s\x1b[0m "

class Displayer(BaseDisplayer):
    '''
    Modified by WAX Mar 19, 2025
    '''
    def __init__(self, N = 7):
        self.dim = N
        self.cellSize = Displayer.half_and_odd(self.dim)
        self.cTemp = "\x1b[%dm%" + str(self.cellSize) + "s\x1b[0m "
        
        if "Windows" == platform.system():
            self.display = self.winDisplay
        else:
            # self.display = self.unixDisplay
            self.display = self.unixDisplayTest

    def display(self, grid):
        pass

    def winDisplay(self, grid):
        
        for i in range(self.dim):
            print("------" * self.dim)
            for j in range(self.dim):
                print("|", end="")
                v = grid.map[int(i)][j]
                if v == -1:
                    string = "x"
                elif v == 0:
                    string = " "
                else:
                    string = str(int(v))
                print("  "+ string + "  ", end="")
            print("|")
        print("------" * self.dim)

    
    def unixDisplay(self, grid):
        
        for i in range(self.dim):
            for j in range(self.dim):
                v = grid.map[int(i)][j]
                if v == 0:
                    string = ""
                elif v == -1:
                    string = "x".center(self.cellSize, " ")
                else:
                    string = str(int(v)).center(self.cellSize, " ")

                print(self.cTemp %(colorMap[v], string), end="")
            print("")
            print("")
        print("")
        print("")

    
    def unixDisplayTest(self, grid):

        for i in range(-1, self.dim):
            print("  ", end = "")
            if i == -1:
                print(" ", end = "")
            else:
                print(i, end = "")
            print("  ", end = "")
            
            for j in range(-1, self.dim):
                if i == -1:
                    if j != -1:
                        # pass
                        coor = str(j).center(self.cellSize, " ")
                        sf = "%" + str(self.cellSize) + "s"                    
                        print(sf % coor, end = " ")
                        # print(" ", end = "")
                    continue
                    
                if j != -1:
                    v = grid.map[int(i)][j]
                    if v == 0:
                        string = ""
                    elif v == -1:
                        string = "x".center(self.cellSize, " ")
                    else:
                        string = str(int(v)).center(self.cellSize, " ")
    
                    print(self.cTemp %(colorMap[v], string), end="")
            print("")
            print("")
        print("")

    
    @staticmethod
    def half_and_odd(num):
        num = num // 2 + 1
        if num % 2 == 0:
            num += 1
            
        return num
        