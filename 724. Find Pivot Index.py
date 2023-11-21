from typing import List

# numbers = [1, 7, 3, 6, 5, 6]
numbers = [2,1,-1]


class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        l = 0
        r = sum(nums)

        for i in range(len(nums)):
            r -= nums[i]

            if l == r:
                return i

            l += nums[i]

        return -1


slt = Solution()
print(slt.pivotIndex(numbers))
