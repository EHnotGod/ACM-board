# P1381 单词背诵
# 24年新星赛-小猪的纸花
from collections import defaultdict
n = int(input())
word = defaultdict(bool)
cnt = defaultdict(int)
len = 0
sum = 0
for i in range(n):
    s1 = input()
    word[s1] = True
m = int(input())
s = [0] * (m + 1)
i = 1
for j in range(1, m + 1):
    s[j] = input()
    if word[s[j]]:
        cnt[s[j]] += 1
    if cnt[s[j]] == 1:
        sum += 1
        len = j - i + 1
    while i <= j:
        if cnt[s[i]] == 1:
            break
        if cnt[s[i]] >= 2:
            cnt[s[i]] -= 1
            i += 1
        if not word[s[i]]:
            i += 1
    len = min(len, j - i + 1)
print(sum)
print(len)