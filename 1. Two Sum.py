class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        numbers = {}

        for i in range(len(nums)):
            if nums[i] in numbers:
                return [numbers[nums[i]], i]
            numbers[target - nums[i]] = i


if __name__ == "__main__":
    slt = Solution()
    assert slt.twoSum([2, 7, 11, 15], 9) == [0, 1]
    assert slt.twoSum([3, 2, 4], 6) == [1, 2]
    assert slt.twoSum([3, 3], 6) == [0, 1]

# Done ✅
