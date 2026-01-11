class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        n = len(s)

        for i in range(1, n // 2 + 1):
            if n % i == 0 and s[:i] * (n // i) == s:
                return True

        return False


if __name__ == "__main__":
    slt = Solution()
    assert slt.repeatedSubstringPattern("abab") == True
    assert slt.repeatedSubstringPattern("aba") == False
    assert slt.repeatedSubstringPattern("abcabcabcabc") == True
