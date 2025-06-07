class Solution:
    def countTriples(self, n: int) -> int:
        if n <= 4:
            return 0

        ans = 0

        for a in range(3, n - 1):
            for b in range(a + 1, n):
                for c in range(a + 2, n + 1):
                    if a ** 2 + b ** 2 == c ** 2:
                        ans += 1

        return ans * 2


if __name__ == '__main__':
    slt = Solution()
    print(slt.countTriples(5))  # 2
    print(slt.countTriples(10))  # 4

# Done ✅
