from typing import List


class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        ans = set([i for i in range(1, len(nums) + 1)]) - set(nums)
        return list(ans)


if __name__ == '__main__':
    slt = Solution()
    print(slt.findDisappearedNumbers([4, 3, 2, 7, 8, 2, 3, 1]))  # [5, 6]
    print(slt.findDisappearedNumbers([1, 1]))  # 2
