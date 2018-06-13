#!/usr/bin/env python
import marshal

f = open("../train")
f2 = open("../test")
fc = open("../fc","w")

d = {}
count = 0
line = f.readline()
# 统计除时间外其他特征出现的次数(只保留出现10次及以上的)，每个特征使用不同的name用来防止不同特征含有相同值导致重复
while True:
    line = f.readline()
    if not line:
        break
    count += 1
    if count % 100000 == 0:
        print count
    lis = line[:-2].split(",")
    for i in xrange(3,len(lis)):
        name = chr(ord('a') + i - 3)
        feat = name + "_" + lis[i]
        if feat in d:
            d[feat] += 1
        else:
            d[feat] = 1

count = 0
line = f2.readline()
while True:
    line = f2.readline()
    if not line:
        break
    count += 1
    if count % 100000 == 0:
        print count
    lis = line[:-2].split(",")
    for i in xrange(2,len(lis)):
        name = chr(ord('a') + i - 2)
        feat = name + "_" + lis[i]
        if feat in d:
            d[feat] += 1
        else:
            d[feat] = 1

s = []
dd = {}
for x in d:
    if d[x] >= 10:
        s.append(x)
marshal.dump(set(s),fc)
