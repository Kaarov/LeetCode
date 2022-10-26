numbers = [1, 7, 3, 6, 5, 6]
# numbers = [2, 1, -1]


class Solution(object):
    def pivotIndex(self, nums):
        left, right = 0, sum(nums)
        for i in range(len(nums)):
            right -= nums[i]
            if left == right:
                return i
            left += nums[i]
        return -1


test = Solution()
print(test.pivotIndex(numbers))

# Done ✅
