from typing import List


class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        sorted_nums = sorted(nums)
        ans = [sorted_nums, list(reversed(sorted_nums))]
        return nums in ans


# nums = [1, 2, 2, 3]
nums = [6, 5, 4, 4]
slt = Solution()
print(slt.isMonotonic(nums))

# Done ✅
