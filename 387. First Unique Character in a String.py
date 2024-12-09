class Solution:
    def firstUniqChar(self, s: str) -> int:
        string = s
        while s:
            if s.count(s[0]) == 1:
                return string.index(s[0])
            s = s.replace(s[0], "")
        return -1


if __name__ == '__main__':
    slt = Solution()
    print(slt.firstUniqChar('loveleetcode'))

# Done ✅
