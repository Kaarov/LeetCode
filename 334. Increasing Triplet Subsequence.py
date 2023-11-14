from typing import List

nums = [2, 1, 5, 0, 4, 6]


class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        i, j = float('inf'), float('inf')
        for k in nums:
            if k <= i:
                i = k
            elif k <= j:
                j = k
            else:
                return True
        return False


slt = Solution()
print(slt.increasingTriplet(nums))

# Done ✅
