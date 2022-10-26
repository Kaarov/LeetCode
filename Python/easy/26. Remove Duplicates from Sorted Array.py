# nums = [1, 1, 2]
nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]


class Solution:
    def removeDuplicates(self, nums):
        a = 1
        for x in range(len(nums) - 1):
            if (nums[x] != nums[x + 1]):
                nums[a] = nums[x + 1]
                a += 1
        return a


a = Solution()
print(a.removeDuplicates(nums))

# Done ✅
