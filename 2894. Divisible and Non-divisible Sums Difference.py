class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        divisible = 0
        not_divisible = 0
        for i in range(1, n + 1):
            if i % m == 0:
                divisible += i
            else:
                not_divisible += i
        return not_divisible - divisible


if __name__ == '__main__':
    slt = Solution()
    print(slt.differenceOfSums(n=10, m=3))  # 19
    print(slt.differenceOfSums(n=5, m=6))  # 15
    print(slt.differenceOfSums(n=5, m=1))  # -15

# Done ✅
