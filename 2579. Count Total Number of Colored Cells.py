class Solution:
    def coloredCells(self, n: int) -> int:
        if n == 1:
            return 1
        ans = 1
        for i in range(2, n + 1):
            ans += 4 * (i - 1)
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.coloredCells(1))  # 1
    print(slt.coloredCells(2))  # 5

# Done ✅
