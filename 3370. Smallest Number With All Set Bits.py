class Solution:
    def smallestNumber(self, n: int) -> int:
        def check_binary(b):
            return all([int(i) for i in b])

        while True:
            if check_binary(format(n, "b")):
                return n
            n += 1


if __name__ == '__main__':
    slt = Solution()
    print(slt.smallestNumber(5))

# Done ✅
