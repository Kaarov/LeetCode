class Solution:
    def romanToInt(self, s: str) -> int:
        check = {
                'IV': 4, 'IX': 9, 'I': 1, 'V': 5, 'XL': 40,
                'XC': 90, 'X': 10, 'L': 50, 'CM': 900, 'CD': 400,
                'C': 100, 'D': 500, 'M': 1000
        }
        ans = 0
        while s:
            for x in check:
                if s[:2] == x:
                    ans += check[x]
                    s = s[2:]
                    break
                elif s[0] == x:
                    ans += check[x]
                    s = s[1:]
                    break
                else:
                    pass
        return ans

# Done ✅
