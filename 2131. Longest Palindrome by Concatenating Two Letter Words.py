from typing import List
from collections import defaultdict


class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        ans = 0
        unmatched = defaultdict(int)
        palindromes = defaultdict(int)

        for word in words:
            if word[0] == word[1]:
                palindromes[word] += 1
                if palindromes[word] == 2:
                    ans += 4
                    del palindromes[word]
                continue
            if word in unmatched:
                ans += 4
                unmatched[word] -= 1
                if not unmatched[word]:
                    del unmatched[word]
                continue
            unmatched[word[::-1]] += 1

        return ans if not palindromes else ans + 2


if __name__ == '__main__':
    slt = Solution()
    print(slt.longestPalindrome(["lc", "cl", "gg"]))  # 6
    print(slt.longestPalindrome(["ab", "ty", "yt", "lc", "cl", "ab"]))  # 8
    print(slt.longestPalindrome(["cc", "ll", "xx"]))  # 2

# Done ✅
