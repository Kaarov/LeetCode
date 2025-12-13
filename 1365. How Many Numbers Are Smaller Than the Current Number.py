from typing import List


# Attempt 1
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        ans = []
        for i in nums:
            count = 0
            for j in nums:
                if i > j:
                    count += 1
            ans.append(count)
        return ans


# Attempt 2
class Solution2:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        numbers = {}

        for i, num in enumerate(sorted(nums)):
            if num not in numbers:
                numbers[num] = i

        return [numbers[num] for num in nums]


if __name__ == '__main__':
    slt = Solution()
    assert slt.smallerNumbersThanCurrent([8, 1, 2, 2, 3]) == [4, 0, 1, 1, 3]
    assert slt.smallerNumbersThanCurrent([6, 5, 4, 8]) == [2, 1, 0, 3]
    assert slt.smallerNumbersThanCurrent([7, 7, 7, 7]) == [0, 0, 0, 0]

    slt2 = Solution2()
    assert slt2.smallerNumbersThanCurrent([8, 1, 2, 2, 3]) == [4, 0, 1, 1, 3]
    assert slt2.smallerNumbersThanCurrent([6, 5, 4, 8]) == [2, 1, 0, 3]
    assert slt2.smallerNumbersThanCurrent([7, 7, 7, 7]) == [0, 0, 0, 0]

# Done ✅
