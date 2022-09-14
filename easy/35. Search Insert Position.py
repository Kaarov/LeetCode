n = [1, 3, 5, 7]
t = 8


class Solution:
    def searchInsert(self, nums, target):
        ans = 0
        if target in nums:
            for x in range(len(nums)):
                if nums[x] == target:
                    ans = x
        else:
            for x in range(len(nums)):
                if nums[x] > target:
                    ans = x
                    break
                else:
                    ans = x + 1
        return ans


a = Solution()
print(a.searchInsert(n, t))
