from typing import List


class Solution:
    def calPoints(self, operations: List[str]) -> int:
        ans = []
        for operation in operations:
            if operation == 'C':
                ans.pop()
            elif operation == 'D':
                ans.append(ans[-1] * 2)
            elif operation == '+':
                ans.append(ans[-1] + ans[-2])
            else:
                ans.append(int(operation))
        return sum(ans)


ops = ["5", "2", "C", "D", "+"]
slt = Solution()
print(slt.calPoints(ops))

# Done ✅
