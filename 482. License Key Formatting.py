class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        ans = []
        s = s.replace("-", "")
        head = len(s) % k

        if head:
            ans.append(s[:head])

        for i in range(head, len(s), k):
            ans.append(s[i:i + k])

        return '-'.join(ans).upper()


if __name__ == "__main__":
    slt = Solution()
    assert slt.licenseKeyFormatting(s="5F3Z-2e-9-w", k=4) == "5F3Z-2E9W"
    assert slt.licenseKeyFormatting(s="2-5g-3-J", k=2) == "2-5G-3J"
    assert slt.licenseKeyFormatting(s="2-4A0r7-4k", k=4) == "24A0-R74K"
    assert slt.licenseKeyFormatting(s="2-4A0r7-4k", k=3) == "24-A0R-74K"

# Done ✅
