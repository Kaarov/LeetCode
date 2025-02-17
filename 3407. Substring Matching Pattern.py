class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        p1, p2 = p.split("*")

        if not p1:
            return p2 in s
        if not p2:
            return p1 in s

        pos = s.find(p1)
        if pos == -1:
            return False

        return p2 in s[pos + len(p1):]


if __name__ == "__main__":
    slt = Solution()
    print(slt.hasMatch(s="leetcode", p="ee*e"))  # True
    print(slt.hasMatch(s="car", p="c*v"))  # False
    print(slt.hasMatch(s="luck", p="u*"))  # True
    print(slt.hasMatch(s="hccc", p="m*c"))  # False

# Done ✅
