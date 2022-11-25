# numbers = [3, 6, 1, 0]
numbers = [0, 0, 0, 1]


class Solution:
    def dominantIndex(self, nums: list[int]) -> int:
        larg = max(nums)
        i = nums.index(larg)
        nums.remove(larg)
        if 2 * max(nums) <= larg:
            return i
        else:
            return -1


slt = Solution()
print(slt.dominantIndex(numbers))

# Done ✅
