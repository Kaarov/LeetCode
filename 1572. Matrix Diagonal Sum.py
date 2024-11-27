from typing import List


class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        length = len(mat)
        ans = 0
        for i in range(length):
            ans += mat[i][i]
            ans += mat[i][length - 1 - i]

        if length % 2:
            ans -= mat[length // 2][length // 2]
        return ans


mat = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]
slt = Solution()
print(slt.diagonalSum(mat))

# Done ✅
