from typing import List


class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        numbers = set(nums)
        for i in numbers:
            if nums.count(i) == 1:
                return i


slt = Solution()
print(slt.singleNumber([4, 1, 2, 1, 2]))  # Expected 4

# Done ✅
