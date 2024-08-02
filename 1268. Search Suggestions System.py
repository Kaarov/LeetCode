from typing import List


class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        ans = []
        products.sort()
        for i in range(1, len(searchWord) + 1):
            count = 0
            result = []
            for j in products:
                if count == 3:
                    break
                elif searchWord[:i] == j[:i]:
                    result.append(j)
                    count += 1
            ans.append(result)
        return ans


products = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
searchWord = "mouse"

slt = Solution()
print(slt.suggestedProducts(products, searchWord))

# Done ✅
