from typing import List

nums = [0, 0, 1]


class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        count = nums.count(0)
        i = 0
        while count != 0:
            if nums[i] == 0:
                count -= 1
                for j in range(i, len(nums) - 1):
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
            else:
                i += 1

# Done ✅
