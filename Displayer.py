"""
Displayer Class module.
Enhanced by WAX.
Fully commented by WAX.
"""
import os
import platform

from BaseDisplayer import BaseDisplayer
from Grid import Grid


colorMap = { 0: 100,
             1: 102,
             2: 105,
             -1: 40}

# cTemp = "\x1b[%dm%7s\x1b[0m "         # no longer needed, implemented via a dynamic resizer in self.cTemp

class Displayer(BaseDisplayer):
    """
    Modified by WAX Mar 19, 2025
    """
    def __init__(self, N = 7):
        self.dim = N
        self.cellSize = Displayer.half_and_odd(self.dim)
        self.cTemp = "\x1b[%dm%" + str(self.cellSize) + "s\x1b[0m "
        
        if "Windows" == platform.system():
            self.display = self.winDisplay
        else:
            # self.display = self.unixDisplay
            self.display = self.unixDisplayNew

    # placeholder for generic display
    def display(self, grid: Grid):
        pass

    def winDisplay(self, grid: Grid) -> None:
        """
        Windows Displayer method.
        Slightly updated.
        """
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

    
    def unixDisplay(self, grid: Grid) -> None:
        """Original Unix Displayer method"""
        
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

    
    def unixDisplayNew(self, grid: Grid) -> None:
        """
        Enhanced Unix Displayer method.
        Shortens the boxes so that the entire grid is closer to a square.
        Displays axes in range(N), where N is dimension.
        """

        print()
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
    def half_and_odd(num: int) -> int:
        """Perform a basic, repeated arithmetic calculation."""
        num = num // 2 + 1
        if num % 2 == 0:
            num += 1
            
        return num
        