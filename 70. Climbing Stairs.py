class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        fibonacci = [1, 1]

        for i in range(n - 1):
            fibonacci.append(fibonacci[-1] + fibonacci[-2])

        return fibonacci[-1]


if __name__ == '__main__':
    slt = Solution()
    print(slt.climbStairs(2))  # 2
    print(slt.climbStairs(3))  # 3

# Done ✅
