class Solution:
    def sumScores(self, s: str) -> int:
        ans = [0] * len(s)
        lo = hi = ii = 0
        for i in range(1, len(s)):
            if i <= hi: ii = i - lo
            if i + ans[ii] <= hi:
                ans[i] = ans[ii]
            else:
                lo, hi = i, max(hi, i)
                while hi < len(s) and s[hi] == s[hi - lo]: hi += 1
                ans[i] = hi - lo
                hi -= 1
        return sum(ans) + len(s)


if __name__ == "__main__":
    slt = Solution()
    assert slt.sumScores("babab") == 9
    assert slt.sumScores("azbazbzaz") == 14
