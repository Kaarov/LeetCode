class Solution:
    def timeRequiredToBuy(self, tickets: list[int], k: int) -> int:
        count = 0
        while tickets:
            ticket = tickets.pop(0)
            if ticket == 1 and k == 0:
                return count + 1
            elif k == 0:
                k = len(tickets)
            else:
                k -= 1

            if ticket > 1:
                tickets.append(ticket - 1)

            count += 1


if __name__ == "__main__":
    slt = Solution()
    assert slt.timeRequiredToBuy([2, 3, 2], 2) == 6
    assert slt.timeRequiredToBuy([5, 1, 1, 1], 0) == 8

# Done ✅
