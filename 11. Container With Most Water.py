from typing import List

height = [1, 8, 6, 2, 5, 4, 8, 3, 7]


class Solution:
    def maxArea(self, h: List[int]) -> int:
        l = 0
        r = len(h) - 1
        ans = 0
        while l < r:
            if h[l] <= h[r]:
                if h[l] * (r - l) > ans:
                    ans = h[l] * (r - l)
                l += 1
            else:
                if h[r] * (r - l) > ans:
                    ans = h[r] * (r - l)
                r -= 1

        return ans


slt = Solution()
print(slt.maxArea(height))

# Done ✅
