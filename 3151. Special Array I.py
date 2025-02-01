from typing import List


class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        if len(nums) < 2:
            return True
        for i in range(len(nums) - 1):
            if (nums[i] + nums[i + 1]) % 2 == 0:
                return False
        return True


if __name__ == "__main__":
    slt = Solution()
    print(slt.isArraySpecial([2, 1, 4]))  # True
    print(slt.isArraySpecial([4, 3, 1, 6]))  # False

# Done ✅
