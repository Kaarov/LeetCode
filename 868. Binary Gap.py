class Solution:
    def binaryGap(self, n: int) -> int:
        ans = 0
        n = bin(n)[2:]
        first = -1

        if n.count("1") < 2:
            return ans

        for i in range(0, len(n)):
            if first != -1 and n[i] == '1':
                ans = max(ans, i - first)
                first = i
            elif first == -1 and n[i] == '1':
                first = i

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.binaryGap(22))  # 2
    print(slt.binaryGap(8))  # 0
    print(slt.binaryGap(5))  # 1

# Done ✅
