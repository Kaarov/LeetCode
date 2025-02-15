from typing import List


class Solution:
    def mostWordsFound(self, sentences: List[str]) -> int:
        ans = 0
        for sentence in sentences:
            ans = max(ans, len(sentence.split()))
        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.mostWordsFound(
        [
            "alice and bob love leetcode",
            "i think so too",
            "this is great thanks very much"
        ]
    ))  # 6
    print(slt.mostWordsFound(
        [
            "please wait",
            "continue to fight",
            "continue to win"
        ]
    ))  # 3

# Done ✅
