from collections import defaultdict
from typing import List


class Solution:
    def oddString(self, words: List[str]) -> str:
        ans = defaultdict(list)
        for word in words:
            res = []
            for i in range(len(word) - 1):
                res.append(ord(word[i]) - ord(word[i + 1]))
            ans[tuple(res)].append(word)

        for key, value in ans.items():
            if len(value) == 1:
                return value[0]


if __name__ == '__main__':
    slt = Solution()
    print(slt.oddString(["adc", "wzy", "abc"]))  # "abc"
    print(slt.oddString(["aaa", "bob", "ccc", "ddd"]))  # "bob"

# Done ✅
