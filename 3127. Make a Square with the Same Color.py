from typing import List


class Solution:
    def canMakeSquare(self, grid: List[List[str]]) -> bool:
        n = len(grid)
        for i in range(n - 1):
            for j in range(n - 1):
                ans = [grid[i][j], grid[i + 1][j], grid[i][j + 1], grid[i + 1][j + 1]]
                color = ans[0]
                if ans.count(color) >= 3 or ans.count(color) <= 1:
                    return True
        return False


if __name__ == '__main__':
    slt = Solution()
    print(slt.canMakeSquare([["B", "W", "B"], ["B", "W", "W"], ["B", "W", "B"]]))  # True
    print(slt.canMakeSquare([["B", "W", "B"], ["W", "B", "W"], ["B", "W", "B"]]))  # False
    print(slt.canMakeSquare([["B", "W", "B"], ["B", "W", "W"], ["B", "W", "W"]]))  # True

# Done ✅
