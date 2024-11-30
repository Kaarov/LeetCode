from typing import List


class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten = 0, 0
        for bill in bills:
            match bill:
                case 5:
                    five += 1
                case 10:
                    if five < 1:
                        return False
                    ten += 1
                    five -= 1
                case _:
                    if ten > 0 and five > 0:
                        ten -= 1
                        five -= 1
                    elif five > 3:
                        five -= 3
                    else:
                        return False
        return True


slt = Solution()
print(slt.lemonadeChange([5, 5, 5, 20]))

# Done ✅
