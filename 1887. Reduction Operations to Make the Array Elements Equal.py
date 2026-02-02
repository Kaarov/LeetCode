from collections import Counter


class Solution:
    def reductionOperations(self, nums: list[int]) -> int:
        counter = Counter(nums)

        ans = 0
        prev = 0

        for num in sorted(counter, reverse=True)[:-1]:
            ans += counter[num]
            ans += prev
            prev += counter[num]

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.reductionOperations([5, 1, 3]) == 3
    assert slt.reductionOperations([1, 1, 1]) == 0
    assert slt.reductionOperations([1, 1, 2, 2, 3]) == 4

# Done ✅
