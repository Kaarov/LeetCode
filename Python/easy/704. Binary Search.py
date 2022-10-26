numbers = [-1, 0, 3, 5, 9, 12]
target_number = 9


class Solution:
    def search(self, nums, target):
        for i in range(len(nums)):
            if nums[i] == target:
                return i
        return -1


test = Solution()
print(test.search(numbers, target_number))

# Done ✅
