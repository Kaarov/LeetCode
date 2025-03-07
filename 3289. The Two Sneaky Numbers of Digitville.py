from typing import List


class Solution:
    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        ans = []
        set_nums = set(nums)
        for i in set_nums:
            if nums.count(i) > 1:
                ans.append(i)
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.getSneakyNumbers([0, 1, 1, 0]))  # [0, 1]
    print(slt.getSneakyNumbers([0, 3, 2, 1, 3, 2]))  # [2, 3]
    print(slt.getSneakyNumbers([7, 1, 5, 4, 3, 4, 6, 0, 9, 5, 8, 2]))  # [4, 5]

# Done ✅
