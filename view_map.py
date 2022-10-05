import numpy as np
import vedo as vd
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import sys


def parse_num(line, i):
    j = 0
    dec = False
    num = ""
    while i + j < len(line):
        if (line[i + j] == "-" and j == 0) or line[i + j].isnumeric():
            num += line[i + j]
        elif line[i + j] == "." and dec == False:
            num += line[i + j]
            dec = True
        elif not line[i + j].isnumeric():
            break
        j += 1
    if num == "-" or num == "." or num == "":
        return None
    else:
        return float(num), i + j

def consume_whitespace(line, i):
    j = 0
    while i + j < len(line):
        if line[i] == " " or line[i] == "\t" or line[i] == "\r":
            j += 1
    return i + j

def parse_map(filename):
    points = []
    f = open(filename, "rb")
    data = f.read(-1)
    lines = data.decode("latin1")
    lines = lines.split("\n")
    print(lines)
    for line in lines:
        i = 0
        nums = []
        while i < len(line):
            if line[i] == " " or line[i] == "\t" or line[i] == "\r":
                i = consume_whitespace(line, i)
            elif line[i] == "#" or line[i] == "\n":
                break
            elif line[i] == "-" or line[i] == "." or line[i].isnumeric():
                num, i = parse_num(line, i)
                nums.append(num)
            else:
                print("Invalid Character")
                print(line[i])
            i += 1
        if len(nums) > 0:
            points.append(nums)
    points = np.array(points)
    points = points.reshape((points.shape[0], 2, 2))
    return points

if len(sys.argv) <= 1:
    print("Too few arguments")
    exit(-1)
filename = sys.argv[1]
m = parse_map(filename)
print(m)
lc = mc.LineCollection(m, linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
plt.show()
