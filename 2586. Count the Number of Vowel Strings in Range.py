from typing import List


class Solution:
    def vowelStrings(self, words: List[str], left: int, right: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        ans = 0
        for i in range(left, right + 1):
            if words[i][0] in vowels and words[i][-1] in vowels:
                ans += 1
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.vowelStrings(words=["are", "amy", "u"], left=0, right=2))  # 2
    print(slt.vowelStrings(words=["hey", "aeo", "mu", "ooo", "artro"], left=1, right=4))  # 3

# Done ✅
