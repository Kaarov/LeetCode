from typing import List


class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        ans = []
        for i in range(len(words)):
            if words[i].find(x) != -1:
                ans.append(i)
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.findWordsContaining(words=["leet", "code"], x="e"))  # [0, 1]
    print(slt.findWordsContaining(words=["abc", "bcd", "aaaa", "cbc"], x="a"))  # [0, 2]
    print(slt.findWordsContaining(words=["abc", "bcd", "aaaa", "cbc"], x="z"))  # []

# Done ✅
