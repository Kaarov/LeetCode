class Solution:
    def canWinNim(self, n: int) -> bool:
        if n % 4 == 0:
            return False
        return True


if __name__ == '__main__':
    slt = Solution()
    print(slt.canWinNim(4))  # False
    print(slt.canWinNim(1))  # True
    print(slt.canWinNim(2))  # True

# Done ✅
