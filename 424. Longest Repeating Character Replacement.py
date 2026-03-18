from collections import defaultdict


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        ans = 0
        left = 0
        count = defaultdict(int)

        for right in range(len(s)):
            count[s[right]] += 1

            while (right - left + 1) - max(count.values()) > k:
                count[s[left]] -= 1
                left += 1

            ans = max(ans, right - left + 1)

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.characterReplacement("ABAB", 2) == 4
    assert slt.characterReplacement("AABABBA", 1) == 4
