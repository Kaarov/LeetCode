class Solution:
    def minOperations(self, s: str) -> int:
        ans = 0

        for i in range(len(s)):
            if i % 2:
                ans += 1 if s[i] == '1' else 0
            else:
                ans += 1 if s[i] == '0' else 0

        return min(ans, len(s) - ans)


if __name__ == '__main__':
    slt = Solution()
    print(slt.minOperations("0100"))  # 1
    print(slt.minOperations("10"))  # 0
    print(slt.minOperations("1111"))  # 2
    print(slt.minOperations("110010"))  # 2
