from typing import List


class Solution:
    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        ans = []
        m = len(grid)
        n = len(grid[0])
        for i in range(n):
            res = 0
            for j in range(m):
                res = max(res, len(str(grid[j][i])))
            ans.append(res)

        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.findColumnWidth([[1], [22], [333]]))  # [3]
    print(slt.findColumnWidth([[-15, 1, 3], [15, 7, 12], [5, 6, -2]]))  # [3, 1, 2]
