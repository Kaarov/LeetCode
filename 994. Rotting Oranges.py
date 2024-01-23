from typing import List


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        queue = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] == 2]
        moves = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        ans = 0
        while queue:
            count = 0
            for i in range(len(queue)):
                x, y = queue.pop(0)
                for dx, dy in moves:
                    if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]):
                        if grid[x + dx][y + dy] == 1:
                            grid[x + dx][y + dy] = 2
                            count += 1
                            queue.append((x + dx, y + dy))
            if count > 0: ans += 1
        check = [(i, j) for i in range(len(grid)) for j in range(len(grid[0])) if grid[i][j] == 1]
        return -1 if check else ans


grid = [
    [2, 1, 1],
    [1, 1, 0],
    [0, 1, 1]]
slt = Solution()
print(slt.orangesRotting(grid))

# Done ✅
