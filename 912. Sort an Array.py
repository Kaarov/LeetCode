class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        min_val, max_val = min(nums), max(nums)
        count = [0] * (max_val - min_val + 1)

        for num in nums:
            count[num - min_val] += 1

        ans = []
        for i in range(len(count)):
            ans.extend([i + min_val] * count[i])

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.sortArray([5, 2, 3, 1]) == [1, 2, 3, 5]
    assert slt.sortArray([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
    assert slt.sortArray([3, -1]) == [-1, 3]

# Note: Could be improved further
