#!/usr/bin/env python3

file = open("testtimes.txt", "r")
sum = 0
count = 0
for ln in file:
	sum += float(ln.split()[3])
	count += 1
print('-----------------------------------')
print('avg time is ' + str(sum/count) + ' milliseconds.')
