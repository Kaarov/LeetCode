class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        ans, left = 0, 0
        char_map = {"a": 0, "b": 0, "c": 0}

        for right in range(0, len(s)):
            char_map[s[right]] += 1

            while char_map["a"] and char_map["b"] and char_map["c"]:
                char_map[s[left]] -= 1
                left += 1

            ans += left

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.numberOfSubstrings("abcabc") == 10
    assert slt.numberOfSubstrings("aaacb") == 3
    assert slt.numberOfSubstrings("abc") == 1
