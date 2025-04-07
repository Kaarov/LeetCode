class Solution:
    def findDelayedArrivalTime(self, arrivalTime: int, delayedTime: int) -> int:
        ans = arrivalTime + delayedTime

        return ans % 24


if __name__ == '__main__':
    slt = Solution()
    print(slt.findDelayedArrivalTime(15, 5))  # 20
    print(slt.findDelayedArrivalTime(13, 11))  # 0

# Done ✅
