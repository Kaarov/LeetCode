from typing import List


class Solution:
    def check(self, nums: List[int]) -> bool:
        sorted_nums = sorted(nums)
        if sorted_nums == nums:
            return True
        for i in range(len(nums)):
            num = nums.pop(0)
            nums.append(num)
            if sorted_nums == nums:
                return True
        return False


if __name__ == "__main__":
    slt = Solution()
    print(slt.check([3, 4, 5, 1, 2]))  # True
    print(slt.check([2, 1, 3, 4]))  # False
    print(slt.check([1, 2, 3]))  # True

# Done ✅
