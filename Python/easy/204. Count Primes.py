number = int(input())


class Solution:
    @staticmethod
    def countPrimes(n):
        # if n < 2:
        #     return 0
        # count = 2
        # ans = 0
        # while count < n:
        #     flag = True
        #     for i in range(2, count):
        #         if (count % i) == 0:
        #             flag = False
        #             break
        #     if flag:
        #         ans += 1
        #     count += 1
        # return count

        if n <= 2:
            return 0
        dp = [True] * n
        dp[0] = dp[1] = False
        for i in range(2, n):
            if dp[i]:
                for j in range(i * i, n, i):
                    dp[j] = False
        return sum(dp)


solution = Solution()
print(solution.countPrimes(number))
