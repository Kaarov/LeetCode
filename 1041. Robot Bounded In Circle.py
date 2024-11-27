class Solution:
    def isRobotBounded(self, instructions: str) -> bool:
        ans = [0, 0]
        direction = 0
        count = 0
        while count < 4:
            for instruction in instructions:
                if instruction == 'R':
                    direction = (direction + 1) % 4
                elif instruction == 'L':
                    direction = (direction - 1) % 4
                else:
                    if direction < 2:
                        ans[(direction + 1) % 2] += 1
                    else:
                        ans[(direction + 1) % 2] -= 1

            if ans == [0, 0]:
                return True
            count += 1
        return False


# instructions = "GGLLGG"
instructions = "GG"
slt = Solution()
print(slt.isRobotBounded(instructions))

# Done ✅
