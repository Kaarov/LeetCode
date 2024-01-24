from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        nums.sort(reverse=True)
        return nums[k - 1]


nums = [3, 2, 1, 5, 6, 4]
k = 2
slt = Solution()
print(slt.findKthLargest(nums, k))
