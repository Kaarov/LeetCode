from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        stack = []
        ans = [0] * len(temperatures)

        for i in range(len(temperatures)):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                index = stack.pop()
                ans[index] = i - index
            stack.append(i)

        return ans


temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
slt = Solution()
print(slt.dailyTemperatures(temperatures))  # [1, 1, 4, 2, 1, 1, 0, 0]
