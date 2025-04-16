class Solution:
    def isThree(self, n: int) -> bool:
        ans = 0
        for i in range(1, n + 1):
            if n % i == 0:
                ans += 1
        return ans == 3


if __name__ == "__main__":
    slt = Solution()
    print(slt.isThree(2))  # False
    print(slt.isThree(4))  # True

# Done ✅
