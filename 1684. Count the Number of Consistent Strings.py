from typing import List


class Solution:
    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        ans = 0
        allowed = set(allowed)
        for word in words:
            if set(word) - allowed == set():
                ans += 1
        return ans


if __name__ == "__main__":
    slt = Solution()
    print(slt.countConsistentStrings(
        allowed="ab",
        words=["ad", "bd", "aaab", "baa", "badab"]
    ))  # 2
    print(slt.countConsistentStrings(
        allowed="abc",
        words=["a", "b", "c", "ab", "ac", "bc", "abc"]
    ))  # 7
    print(slt.countConsistentStrings(
        allowed="cad",
        words=["cc", "acd", "b", "ba", "bac", "bad", "ac", "d"]
    ))  # 4

# Done ✅
