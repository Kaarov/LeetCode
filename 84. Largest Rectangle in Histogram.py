class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        ans = 0
        stack = []

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                start, height = stack.pop()
                ans = max(ans, height * (i - start))
            stack.append((start, h))
        for i, h in stack:
            ans = max(ans, h * (len(heights) - i))
        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.largestRectangleArea([2, 1, 5, 6, 2, 3]) == 10
    assert slt.largestRectangleArea([2, 4]) == 4

# Done ✅
