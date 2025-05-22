from typing import List


class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        ans = []
        nums.sort()

        while nums:
            ans.append((nums.pop(0) + nums.pop(-1)) / 2)

        return min(ans)


if __name__ == '__main__':
    slt = Solution()
    print(slt.minimumAverage([7, 8, 3, 4, 15, 13, 4, 1]))  # 5.5
    print(slt.minimumAverage([1, 9, 8, 3, 10, 5]))  # 5.5
    print(slt.minimumAverage([1, 2, 3, 7, 8, 9]))  # 5.0

# Done ✅
