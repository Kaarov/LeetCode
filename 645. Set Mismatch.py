class Solution:
    def findErrorNums(self, nums: list[int]) -> list[int]:
        numbers = [0 for i in range(len(nums))]

        for i in nums:
            numbers[i - 1] += 1

        return [numbers.index(2) + 1, numbers.index(0) + 1]


if __name__ == '__main__':
    slt = Solution()
    print(slt.findErrorNums([1, 2, 2, 4]))  # [2, 3]
    print(slt.findErrorNums([1, 1]))  # [1, 2]

# Done ✅
