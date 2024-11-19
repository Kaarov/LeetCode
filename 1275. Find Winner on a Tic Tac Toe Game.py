from typing import List


class Solution:
    def tictactoe(self, moves: List[List[int]]) -> str:
        ans = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        count = 1
        for i in moves:
            ans[i[0]][i[1]] = "X" if count % 2 else "O"
            count += 1
        if (
                ans[0][0] == ans[1][1] == ans[2][2] == "X"
                or ans[0][2] == ans[1][1] == ans[2][0] == "X"
                or ans[0][0] == ans[0][1] == ans[0][2] == "X"
                or ans[1][0] == ans[1][1] == ans[1][2] == "X"
                or ans[2][0] == ans[2][1] == ans[2][2] == "X"
                or ans[0][0] == ans[1][0] == ans[2][0] == "X"
                or ans[0][1] == ans[1][1] == ans[2][1] == "X"
                or ans[0][2] == ans[1][2] == ans[2][2] == "X"
        ):
            return "A"
        elif (
                ans[0][0] == ans[1][1] == ans[2][2] == "O"
                or ans[0][2] == ans[1][1] == ans[2][0] == "O"
                or ans[0][0] == ans[0][1] == ans[0][2] == "O"
                or ans[1][0] == ans[1][1] == ans[1][2] == "O"
                or ans[2][0] == ans[2][1] == ans[2][2] == "O"
                or ans[0][0] == ans[1][0] == ans[2][0] == "O"
                or ans[0][1] == ans[1][1] == ans[2][1] == "O"
                or ans[0][2] == ans[1][2] == ans[2][2] == "O"
        ):
            return "B"
        elif any([ans[0].count(0), ans[1].count(0), ans[2].count(0)]):
            return "Pending"
        else:
            return "Draw"


# moves = [[0, 0], [2, 0], [1, 1], [2, 1], [2, 2]]
moves = [[2, 0], [1, 1], [0, 2], [2, 1], [1, 2], [1, 0], [0, 0], [0, 1]]
slt = Solution()
print(slt.tictactoe(moves))

# Done ✅
