class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums * 2


if __name__ == '__main__':
    slt = Solution()
    slt.getConcatenation(nums=[1, 2, 1])  # [1, 2, 1, 1, 2, 1]
    slt.getConcatenation(nums=[1, 3, 2, 1])  # [1, 3, 2, 1, 1, 3, 2, 1]

# Done ✅
