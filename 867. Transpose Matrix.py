from typing import List


class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        col = len(matrix[0])
        row = len(matrix)

        ans = [[0] * row for _ in range(col)]

        for i in range(row):
            for j in range(col):
                ans[j][i] = matrix[i][j]

        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    print(slt.transpose([[1, 2, 3], [4, 5, 6]]))  # [[1, 4], [2, 5], [3, 6]]

# Done ✅
