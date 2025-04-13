class Solution:
    def countGoodNumbers(self, n: int) -> int:
        mod = 10 ** 9 + 7

        odd = n // 2
        even = n - odd

        return (pow(5, even, mod) * pow(4, odd, mod)) % mod


if __name__ == '__main__':
    slt = Solution()
    print(slt.countGoodNumbers(1))  # 5
    print(slt.countGoodNumbers(4))  # 400
    print(slt.countGoodNumbers(50))  # 564908303
