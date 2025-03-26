from typing import List
from collections import defaultdict


class Solution:
    def modifiedMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        ans = defaultdict(list)
        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                ans[i].append(matrix[j][i])
        ans = {key: max(value) for key, value in ans.items()}

        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                if matrix[j][i] == -1:
                    matrix[j][i] = ans[i]

        return matrix


if __name__ == "__main__":
    slt = Solution()
    print(slt.modifiedMatrix(matrix=[[1, 2, -1], [4, -1, 6], [7, 8, 9]]))  # [[1, 2, 9], [4, 8, 6], [7, 8, 9]]
    print(slt.modifiedMatrix(matrix=[[3, -1], [5, 2]]))  # [[3, 2], [5, 2]]

# Done ✅
