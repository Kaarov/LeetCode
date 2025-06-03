class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def symmetric(num: int) -> bool:
            str_num = str(num)
            length = len(str_num)

            if length % 2:
                return False

            list_num = list(map(int, str_num))

            return sum(list_num[:length // 2]) == sum(list_num[length // 2:])

        ans = 0

        for i in range(low, high + 1):
            if symmetric(i):
                ans += 1

        return ans


if __name__ == '__main__':
    slt = Solution()
    print(slt.countSymmetricIntegers(low=1, high=100))  # 9
    print(slt.countSymmetricIntegers(low=1200, high=1230))  # 4

# Done ✅
