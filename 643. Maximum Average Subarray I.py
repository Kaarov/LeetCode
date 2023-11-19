from typing import List

numbers = [1, 12, -5, -6, 50, 3]
key = 4


class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        currSum = ans = sum(nums[:k])
        for i in range(len(nums) - k):
            currSum = currSum - nums[i] + nums[i + k]
            ans = max(ans, currSum)
        return ans / k


slt = Solution()
print(slt.findMaxAverage(numbers, key))
