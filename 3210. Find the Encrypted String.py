class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        idx = k % len(s)
        return s[idx:] + s[:idx]


if __name__ == '__main__':
    slt = Solution()
    print(slt.getEncryptedString(s="dart", k=3))  # "tdar"
    print(slt.getEncryptedString(s="aaa", k=1))  # "aaa"

# Done ✅
