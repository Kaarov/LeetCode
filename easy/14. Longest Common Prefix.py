strs = ["flower", "flow", "flight"]

longest = ''
for x in strs:
    if len(x) > len(longest):
        longest = x
strs.remove(longest)
check = ''
for x in strs:
    for y in range(len(longest)):
        if longest[y] == x[y]:
            check 
