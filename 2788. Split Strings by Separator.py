from typing import List


class Solution:
    def splitWordsBySeparator(self, words: List[str], separator: str) -> List[str]:
        ans = []
        for word in words:
            word = word.split(separator)
            ans.extend([w for w in word if w is not ""])
        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.splitWordsBySeparator(
        words=["one.two.three", "four.five", "six"],
        separator="."
    ))  # ["one", "two", "three", "four", "five", "six"]
    print(slt.splitWordsBySeparator(words=["$easy$", "$problem$"], separator="$"))  # ["easy", "problem"]
    print(slt.splitWordsBySeparator(words=["|||"], separator="|"))  # []

# Done ✅
