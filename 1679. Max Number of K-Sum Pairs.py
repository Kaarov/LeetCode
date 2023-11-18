from typing import List

nums = [4, 1, 3, 1, 3, 2, 5, 1, 5, 2, 1, 5, 4]
k = 2
# nums = [1, 2, 3, 4]
# k = 5


class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        l, r = 0, len(nums) - 1
        res = 0
        while l < r:
            summ = nums[l] + nums[r]
            if (summ == k):
                l += 1
                r -= 1
                res += 1
            elif (summ < k):
                l += 1
            else:
                r -= 1
        return res


slt = Solution()
print(slt.maxOperations(nums, k))
