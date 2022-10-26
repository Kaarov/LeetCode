# sym = "ab#c"
# tab = "ad#c"
# sym = "ab##"
# tab = "c#d#"
sym = "a##c"
tab = "#a#c"


class Solution:
    def backspaceCompare(self, s, t):
        new_s = new_t = ""
        for letter in s:
            if letter != '#':
                new_s += letter
            else:
                new_s = self.remove_previous(new_s)
        for letter in t:
            if letter != '#':
                new_t += letter
            else:
                new_t = self.remove_previous(new_t)

        if new_s == new_t:
            return True
        return False

    def remove_previous(self, s):
        if s == "":
            return s
        return s[:-1]


slt = Solution()
print(slt.backspaceCompare(sym, tab))
