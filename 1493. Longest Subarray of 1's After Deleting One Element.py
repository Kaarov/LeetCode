from typing import List

nums = [0, 1, 1, 1, 0, 1, 1, 0, 1]


class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        zeroCount = 1
        i = j = 0

        for j in range(len(nums)):
            zeroCount -= nums[j] == 0

            if zeroCount < 0:
                zeroCount += nums[i] == 0
                i += 1

        return j - i


slt = Solution()
print(slt.longestSubarray(nums))
