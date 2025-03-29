class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        def check_bit(bit: str) -> bool:
            for i in range(len(bit) - 1):
                if bit[i] == bit[i + 1]:
                    return False
            return True

        ans = check_bit(format(n, 'b'))
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.hasAlternatingBits(5))  # True
    print(slt.hasAlternatingBits(7))  # False
    print(slt.hasAlternatingBits(11))  # False

# Done ✅
