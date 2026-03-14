class Solution:
    def magicalString(self, n: int) -> int:
        res = '122'
        i = 2
        while len(res) < n:
            if res[i] == '2':
                if res[-1] == '1':
                    res += '2' * 2
                else:
                    res += '1' * 2
            else:
                if res[-1] == '2':
                    res += '1'
                else:
                    res += '2'
            i += 1
        if len(res) > n:
            res = res[:n]
        return res.count('1')


if __name__ == "__main__":
    slt = Solution()
    assert slt.magicalString(6) == 3
    assert slt.magicalString(1) == 1
