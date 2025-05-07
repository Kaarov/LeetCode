from typing import List


class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        ans = 0
        for i in nums:
            if not i % 2:
                ans += 1

            if ans >= 2:
                return True

        return False


if __name__ == "__main__":
    slt = Solution()
    print(slt.hasTrailingZeros([1, 2, 3, 4, 5]))  # True
    print(slt.hasTrailingZeros([2, 4, 8, 16]))  # True
    print(slt.hasTrailingZeros([1, 3, 5, 7, 9]))  # False
