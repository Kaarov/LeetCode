from typing import List

grid = [[3, 1, 2, 2], [1, 4, 4, 5], [2, 4, 2, 2], [2, 4, 2, 2]]


class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        count = 0
        ans = {}
        n = len(grid)

        for i in grid:
            i = tuple(i)
            ans[i] = ans.get(i, 0) + 1

        for i in range(n):
            nest_list = []
            for j in range(n):
                nest_list.append(grid[j][i])
            nest_list = tuple(nest_list)
            count += ans.get(nest_list, 0)

        return count


slt = Solution()
print(slt.equalPairs(grid))
