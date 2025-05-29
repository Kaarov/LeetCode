from typing import List


class Solution:
    def circularGameLosers(self, n: int, k: int) -> List[int]:
        friends = [1, ]
        losers = list(range(2, n + 1))

        count = 0

        while True:
            count += k
            ball = friends[-1] + count
            ans = ball % n if ball % n != 0 else n
            if ans in losers:
                losers.remove(ans)
            if ans in friends:
                return losers
            friends.append(ans)


if __name__ == "__main__":
    slt = Solution()
    print(slt.circularGameLosers(n=5, k=2))  # [4, 5]
    print(slt.circularGameLosers(n=4, k=4))  # [2, 3, 4]

# Done ✅
