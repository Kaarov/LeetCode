from typing import List


class Solution:
    def checkXMatrix(self, grid: List[List[int]]) -> bool:
        length = len(grid)
        for i in range(length):
            for j in range(length):
                if i == j or i + j == length - 1:
                    if grid[i][j] == 0:
                        return False
                else:
                    if grid[i][j] != 0:
                        return False
        return True


if __name__ == '__main__':
    slt = Solution()
    print(slt.checkXMatrix([[2, 0, 0, 1], [0, 3, 1, 0], [0, 5, 2, 0], [4, 0, 0, 2]]))  # True
    print(slt.checkXMatrix([[5, 7, 0], [0, 3, 1], [0, 5, 0]]))  # False

# Done ✅
