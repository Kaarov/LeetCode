class Solution:
    def dailyTemperatures(self, temperatures: list[int]) -> list[int]:
        ans = [0] * len(temperatures)
        stack = []

        for i in range(len(temperatures)):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                num = stack.pop()
                ans[num] = i - num

            stack.append(i)

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert slt.dailyTemperatures([30, 40, 50, 60]) == [1, 1, 1, 0]
    assert slt.dailyTemperatures([30, 60, 90]) == [1, 1, 0]
