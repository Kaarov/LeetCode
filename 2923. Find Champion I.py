from typing import List


class Solution:
    def findChampion(self, grid: List[List[int]]) -> int:
        ans, summa = 0, 0
        for i in range(len(grid)):
            if sum(grid[i]) > summa:
                summa = sum(grid[i])
                ans = i
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.findChampion([[0, 1], [0, 0]]))  # 0
    print(slt.findChampion([[0, 0, 1], [1, 0, 1], [0, 0, 0]]))  # 1
    print(slt.findChampion([[0, 1, 1], [0, 0, 0], [0, 1, 0]]))  # 0

# Done ✅
