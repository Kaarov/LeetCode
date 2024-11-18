class Solution:
    def judgeCircle(self, moves: str) -> bool:
        ans = [0, 0]  # [Left/Right, Up/Down]
        for move in moves:
            match move:
                case "L":
                    ans[0] += 1
                case "R":
                    ans[0] -= 1
                case "U":
                    ans[1] += 1
                case "D":
                    ans[1] -= 1
        return True if ans == [0, 0] else False


moves = "UD"
slt = Solution()
print(slt.judgeCircle(moves))

# Done ✅
