from typing import List

nums = [1, 2, 3, 4]


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        prefix = [1, ]
        suffix = [1, ]
        ans = []

        length = len(nums)

        for i in range(length - 1):
            prefix.append(prefix[i] * nums[i])

        for i in range(1, length):
            suffix.append(suffix[i - 1] * nums[-i])

        suffix = suffix[::-1]
        for i in range(length):
            ans.append(prefix[i] * suffix[i])

        return ans


slt = Solution()
print(slt.productExceptSelf(nums))
