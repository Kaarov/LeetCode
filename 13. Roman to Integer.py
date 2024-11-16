class Solution:
    def romanToInt(self, s: str) -> int:
        roman = {
            'IV': 4, 'IX': 9, 'I': 1, 'V': 5, 'XL': 40,
            'XC': 90, 'X': 10, 'L': 50, 'CM': 900, 'CD': 400,
            'C': 100, 'D': 500, 'M': 1000
        }
        ans = 0
        while s:
            idx = 2 if len(s) > 1 and s[:2] in roman else 1
            ans += roman[s[:idx]]
            s = s[idx:]
        return ans


s = "LVIII"
slt = Solution()
print(slt.romanToInt(s))

# Done ✅
