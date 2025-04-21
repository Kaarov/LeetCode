from typing import List


class Solution:
    def findIndices(self, nums: List[int], indexDifference: int, valueDifference: int) -> List[int]:
        ans = [-1, -1]

        for i in range(len(nums)):
            for j in range(len(nums)):
                if abs(i - j) >= indexDifference and abs(nums[i] - nums[j]) >= valueDifference:
                    return [i, j]

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.findIndices(
        nums=[5, 1, 4, 1],
        indexDifference=2,
        valueDifference=4,
    ))  # [0, 3]
    print(slt.findIndices(
        nums=[2, 1],
        indexDifference=0,
        valueDifference=0,
    ))  # [0, 0]
    print(slt.findIndices(
        nums=[1, 2, 3],
        indexDifference=2,
        valueDifference=4,
    ))  # [-1, -1]

# Done ✅
