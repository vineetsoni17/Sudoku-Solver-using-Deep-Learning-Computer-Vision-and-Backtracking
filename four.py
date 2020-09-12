import numpy as np
import os

def path(grid, n):
    for position, i in np.ndenumerate(grid):
        if i==0:
            for j in range(1,10):
                if not check(grid, j, position, n):
                    if j==9:
                        break
                    else:
                        continue
                else:
                    break
            if check_indi(grid, j, position) and position!=n:
                return True
            elif not check_indi(grid, j, position):
                grid[position]=0
                return False
            elif position==n:
                return grid  
        if position==(8,8):
            answer(grid)
            return True 

def check(grid, j, position, n):
    for x in grid[position[0], :]:
        if j==x:
            grid[position]=0
            return False
    for y in grid[:, position[1]]:
        if j==y:
            grid[position]=0
            return False
    m=(position[0])//3
    p=(position[1])//3
    for z in np.nditer(grid[3*m:3*m+3, 3*p:3*p+3]):
        if j==z:
            grid[position]=0
            return False
    grid[position]=j
    return path(grid, n) 

def check_indi(grid, j, position):
    for x in grid[position[0], :]:
        if j==x:
            return False
    for y in (grid[:, position[1]]):
        if j==y:
            return False
    m=(position[0])//3
    p=(position[1])//3
    for z in np.nditer(grid[3*m:3*m+3, 3*p:3*p+3]):
        if j==z:
            return False
    return True

def answer(grid):
    print("Solution :\n", grid)

def main(grid):
    for p, x in np.ndenumerate(grid):
        if(x==0):
            n=p
            break
    print("Question : " ,'\n', grid,)
    path(grid, n)

# grid = np.array()
# main(grid)
