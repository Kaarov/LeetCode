from typing import List


class Solution:
    def satisfiesConditions(self, grid: List[List[int]]) -> bool:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i + 1 < len(grid) and j < len(grid[0]) and grid[i][j] != grid[i + 1][j]:
                    return False
                if i < len(grid) and j + 1 < len(grid[0]) and grid[i][j] == grid[i][j + 1]:
                    return False
        return True


if __name__ == "__main__":
    slt = Solution()
    print(slt.satisfiesConditions([[1, 0, 2], [1, 0, 2]]))  # True
    print(slt.satisfiesConditions([[1, 1, 1], [0, 0, 0]]))  # False
    print(slt.satisfiesConditions([[1], [2], [3]]))  # False

# Done ✅
