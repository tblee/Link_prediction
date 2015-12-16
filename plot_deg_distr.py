from collections import Counter
import matplotlib.pyplot as plt

deg = [0] * 1000
with open("repos/1000_repos/snapshot-0630.txt") as f:
    for line in f:
        parts = line.rstrip().split(',')
        n1, n2 = int(parts[0]), int(parts[1])
        deg[n1] += 1
        deg[n2] += 1

deg_distr = sorted(Counter(deg).most_common(), key=lambda t: t[0])
deg_value_count = len(deg_distr)

cumul_deg_distr = [None] * deg_value_count
cumul_deg_distr[deg_value_count-1] = deg_distr[deg_value_count-1]
for i in range(deg_value_count-2, -1, -1):
    cumul_deg_distr[i] = (deg_distr[i][0], cumul_deg_distr[i+1][1] + deg_distr[i][1])

x = [t[0] for t in cumul_deg_distr]
y = [t[1] for t in cumul_deg_distr]

plt.plot(x, y)
plt.yscale('log', nonposy='clip')
plt.xlabel("Node degree")
plt.ylabel("Cumulative frequency")
plt.savefig("deg_distr.png")
