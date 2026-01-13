class Solution:
    def repeatedStringMatch(self, a: str, b: str) -> int:
        goal = a
        count = 1

        while len(goal) < len(b):
            goal += a
            count += 1

        if b in goal: return count
        if b in goal + a: return count + 1

        return -1


if __name__ == "__main__":
    slt = Solution()
    assert slt.repeatedStringMatch("abcd", "cdabcdab") == 3
    assert slt.repeatedStringMatch("a", "aa") == 2
    assert slt.repeatedStringMatch("a", "a") == 1
    assert slt.repeatedStringMatch("abc", "wxyz") == -1
    assert slt.repeatedStringMatch("abc", "cabcabca") == 4
    assert slt.repeatedStringMatch("aaaaaaaaaaaaaaaaaaaaaab", "ba") == 2

# Done ✅
# Note: Could be improved further
