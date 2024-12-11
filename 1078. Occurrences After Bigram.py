from typing import List


class Solution:
    def findOcurrences(self, text: str, first: str, second: str) -> List[str]:
        ans = []
        text = text.split()
        for idx, word in enumerate(text):
            if idx >= 2 and text[idx - 2] == first and text[idx - 1] == second:
                ans.append(word)
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.findOcurrences(
        text="jkypmsxd jkypmsxd kcyxdfnoa jkypmsxd kcyxdfnoa jkypmsxd kcyxdfnoa kcyxdfnoa jkypmsxd kcyxdfnoa",
        first="kcyxdfnoa",
        second="jkypmsxd"
    ))

# Done ✅
