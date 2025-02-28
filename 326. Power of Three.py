class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        if n <= 0:
            return False
        count = 0
        while True:
            ans = 3 ** count
            if ans == n:
                return True
            if ans > n:
                return False
            count += 1


if __name__ == "__main__":
    slt = Solution()
    print(slt.isPowerOfThree(27))  # True
    print(slt.isPowerOfThree(0))  # False
    print(slt.isPowerOfThree(-1))  # False

# Done ✅
