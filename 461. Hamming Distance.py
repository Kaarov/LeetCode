class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')


if __name__ == "__main__":
    slt = Solution()
    print(slt.hammingDistance(1, 4))  # 2
    print(slt.hammingDistance(3, 1))  # 1

# Done ✅
