from typing import List


class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        row = set()
        column = set()
        for i in range(len(matrix)):
            if matrix[i].count(0):
                for j in range(len(matrix[0])):
                    if matrix[i][j] == 0:
                        column.add(j)
                row.add(i)

        if row:
            for i in range(len(matrix)):
                for j in range(len(matrix[0])):
                    if i in row or j in column:
                        matrix[i][j] = 0


slt = Solution()
slt.setZeroes([[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]])

# Done ✅
