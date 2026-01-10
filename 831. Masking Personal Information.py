class Solution:
    def maskPII(self, s: str) -> str:
        def maks_email(email: str) -> str:
            name, domain = email.split("@")
            ans = name[0] + "*****" + name[-1] + "@" + domain
            return ans.lower()

        def mask_phone(phone: str) -> str:
            phone = "".join([p for p in phone if p.isdigit()])

            country_code = {
                0: "***-***-",
                1: "+*-***-***-",
                2: "+**-***-***-",
                3: "+***-***-***-",
            }

            return country_code[len(phone) - 10] + phone[-4:]

        return maks_email(s) if "@" in s else mask_phone(s)


if __name__ == "__main__":
    slt = Solution()
    assert slt.maskPII("LeetCode@LeetCode.com") == "l*****e@leetcode.com"
    assert slt.maskPII("AB@qq.com") == "a*****b@qq.com"
    assert slt.maskPII("1(234)567-890") == "***-***-7890"

# Done ✅
# Note: Could be improved further
