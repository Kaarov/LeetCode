import string

from typing import List
from collections import Counter


class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        for punctuation in string.punctuation:
            paragraph = paragraph.replace(punctuation, " ")

        ans = Counter(paragraph.lower().split()).most_common()

        for word, count in ans:
            if word not in banned:
                return word


if __name__ == '__main__':
    slt = Solution()
    print(slt.mostCommonWord(
        paragraph="Bob hit a ball, the hit BALL flew far after it was hit.",
        banned=["hit"]
    ))  # "ball"
    print(slt.mostCommonWord(
        paragraph="a.",
        banned=[]
    ))  # a

# Done ✅
