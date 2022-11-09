numbers = [2, 0, 2, 1, 1, 0]


class Solution:
    def sortColors(self, nums: List[int]) -> None:
        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] > nums[j]:
                    k = nums[i]
                    nums[i] = nums[j]
                    nums[j] = k


slt = Solution()
print(slt.sortColors(numbers))

# Done ✅