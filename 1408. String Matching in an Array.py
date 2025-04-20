from typing import List


class Solution:
    def stringMatching(self, words: List[str]) -> List[str]:
        ans = []
        for i in range(len(words)):
            for j in range(len(words)):
                if words[i] in words[j] and i != j:
                    ans.append(words[i])
                    break

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.stringMatching(["mass", "as", "hero", "superhero"]))  # ["as", "hero"]
    print(slt.stringMatching(["leetcode", "et", "code"]))  # ["et", "code"]
    print(slt.stringMatching(["blue", "green", "bu"]))  # []

# Done ✅
