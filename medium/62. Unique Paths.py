m_number = 3
n_number = 2


class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = []
        for i in range(m):
            dpl = []
            for j in range(n):
                dpl.append(0)
            dp.append(dpl)

        # dp = [[0 for j in range(n)] for i in range(m)]
        dp[-1][-1] = 1

        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i > 0:
                    dp[i - 1][j] += dp[i][j]
                if j > 0:
                    dp[i][j - 1] += dp[i][j]

        return dp[0][0]


slt = Solution()
print(slt.uniquePaths(m_number, n_number))
