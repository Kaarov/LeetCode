numbers = [1, 2, 3, 1, 1, 3]


class Solution:
    def numIdenticalPairs(self, nums) -> int:
        ans = 0
        for i in range(len(nums)-1):
            for j in range(i+1, len(nums)):
                if nums[i] == nums[j]:
                    ans += 1
        return ans


slt = Solution()
print(slt.numIdenticalPairs(numbers))
