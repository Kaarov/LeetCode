from typing import List

nums = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0]
k = 2


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        ans = zeroCount = i = 0

        for j in range(len(nums)):
            if nums[j] == 0:
                zeroCount += 1

            while zeroCount > k:
                if nums[i] == 0:
                    zeroCount -= 1
                i += 1
            ans = max(ans, j - i + 1)

        return ans


slt = Solution()
print(slt.longestOnes(nums, k))
