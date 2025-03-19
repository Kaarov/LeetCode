from typing import List
from collections import Counter


class Solution:
    def removeAnagrams(self, words: List[str]) -> List[str]:
        i = 0
        while i < len(words) - 1:
            if Counter(words[i]) == Counter(words[i + 1]):
                del words[i + 1]
            else:
                i += 1
        return words


if __name__ == '__main__':
    slt = Solution()
    print(slt.removeAnagrams(["abba", "baba", "bbaa", "cd", "cd"]))  # ["abba", "cd"]
    print(slt.removeAnagrams(["a", "b", "c", "d", "e"]))  # ["a", "b", "c", "d", "e"]

# Done ✅
