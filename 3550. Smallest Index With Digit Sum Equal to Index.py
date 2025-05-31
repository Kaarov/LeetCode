from typing import List


class Solution:
    def smallestIndex(self, nums: List[int]) -> int:
        def equalToIndex(num: int, index: int) -> bool:
            return index == sum(map(int, str(num)))

        ans = [i for i in range(len(nums)) if equalToIndex(nums[i], i)]
        if ans:
            return min(ans)
        return -1


if __name__ == '__main__':
    slt = Solution()
    print(slt.smallestIndex([1, 3, 2]))  # 2
    print(slt.smallestIndex([1, 10, 11]))  # 1
    print(slt.smallestIndex([1, 2, 3]))  # -1

# Done ✅
