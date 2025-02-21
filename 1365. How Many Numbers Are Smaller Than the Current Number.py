from typing import List


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


if __name__ == '__main__':
    slt = Solution()
    print(slt.smallerNumbersThanCurrent([8, 1, 2, 2, 3]))  # [4, 0, 1, 1, 3]
    print(slt.smallerNumbersThanCurrent([6, 5, 4, 8]))  # [2, 1, 0, 3]
    print(slt.smallerNumbersThanCurrent([7, 7, 7, 7]))  # [0, 0, 0, 0]

# Done ✅
