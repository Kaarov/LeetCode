from typing import List


class Solution:
    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        item_dict = {
            "type": 0,
            "color": 1,
            "name": 2,
        }
        ruleKey = item_dict[ruleKey]

        ans = [item for item in items if item[ruleKey] == ruleValue]

        return len(ans)


if __name__ == "__main__":
    slt = Solution()
    print(slt.countMatches(
        items=[
            ["phone", "blue", "pixel"],
            ["computer", "silver", "lenovo"],
            ["phone", "gold", "iphone"]
        ],
        ruleKey="color",
        ruleValue="silver"
    ))  # 1
    print(slt.countMatches(
        items=[
            ["phone", "blue", "pixel"],
            ["computer", "silver", "phone"],
            ["phone", "gold", "iphone"]
        ],
        ruleKey="type",
        ruleValue="phone"
    ))  # 2

# Done ✅
