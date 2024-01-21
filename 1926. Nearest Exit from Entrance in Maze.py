from collections import deque
from typing import List


class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        deq = deque([(entrance[0], entrance[1], 0)])
        maze[entrance[0]][entrance[1]] = '+'
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while deq:
            i, j, steps = deq.popleft()
            if i == 0 or i == len(maze) - 1 or j == 0 or j == len(maze[0]) - 1:
                if steps > 0:
                    return steps
            for dx, dy in directions:
                if 0 <= i + dx < len(maze) and 0 <= j + dy < len(maze[0]) and maze[i + dx][j + dy] == '.':
                    deq.append([i + dx, j + dy, steps + 1])
                    maze[i + dx][j + dy] = '+'
        return -1


slt = Solution()
maze = [["+", "+", ".", "+"], [".", ".", ".", "+"], ["+", "+", "+", "."]]
entrance = [1, 2]
print(slt.nearestExit(maze, entrance))
