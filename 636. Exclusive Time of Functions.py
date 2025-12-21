class Solution:
    def exclusiveTime(self, n: int, logs: list[str]) -> list[int]:
        ans = [0] * n

        stack = []
        prev_start_time = 0

        for log in logs:
            func_id, call_type, timestamp = log.split(":")

            func_id = int(func_id)
            timestamp = int(timestamp)

            if call_type == "start":
                if stack:
                    ans[stack[-1]] += timestamp - prev_start_time

                stack.append(func_id)
                prev_start_time = timestamp
            else:
                ans[stack.pop()] += timestamp - prev_start_time + 1
                prev_start_time = timestamp + 1

        return ans


if __name__ == '__main__':
    slt = Solution()
    assert slt.exclusiveTime(
        n=2,
        logs=["0:start:0", "1:start:2", "1:end:5", "0:end:6"],
    ) == [3, 4]
    assert slt.exclusiveTime(
        n=1,
        logs=["0:start:0", "0:start:2", "0:end:5", "0:start:6", "0:end:6", "0:end:7"],
    ) == [8]
    assert slt.exclusiveTime(
        n=2,
        logs=["0:start:0", "0:start:2", "0:end:5", "1:start:6", "1:end:6", "0:end:7"],
    ) == [7, 1]
