class Solution:
    def largestAltitude(self, gain: list[int]) -> int:
        ans = 0
        sum = 0
        for i in gain:
            sum += i
            ans = max(ans, sum)

        return ans


if __name__ == "__main__":
    slt = Solution()
    assert slt.largestAltitude([-5, 1, 5, 0, -7]) == 1
    assert slt.largestAltitude([-4, -3, -2, -1, 4, 3, 2]) == 0

# Done ✅
