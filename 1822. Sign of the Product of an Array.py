from typing import List


class Solution:
    def arraySign(self, nums: List[int]) -> int:
        ans = 0
        for num in nums:
            if num == 0:
                return 0
            ans += (1 if num < 0 else 0)
        return -1 if ans % 2 else 1


nums = [-1, -2, -3, -4, 3, 2, 1]
slt = Solution()
print(slt.arraySign(nums))

# Done ✅
