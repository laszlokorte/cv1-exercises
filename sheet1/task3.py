# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import math
import os

print(os.path.basename(__file__))
print("".join(open(__file__, 'r').readlines()[0:3]))

hist1 = [0.3,0.1,0.2,0,0.4,0]
hist2 = [0,0.1,0.3,0,0.2,0.4]
print(f"p(X) = {hist1}")
print(f"q(X) = {hist2}")

exp1 = sum([p*x for x,p in enumerate(hist1)])
sum1 = " + ".join([f"{p}*{x}" for x,p in enumerate(hist1)])
print(f"E[p(X)] = {sum1} = {exp1}")
exp2 = sum([p*x for x,p in enumerate(hist2)])
sum2 = " + ".join([f"{p}*{x}" for x,p in enumerate(hist2)])
print(f"E[q(X)] = {sum2} = {exp2}")

comu1 = [0.3,0.4,0.6,0.6,1.0,1.0]
comu2 = [0.0,0.1,0.4,0.4,0.6,1.0]

print(f"P(X) = {comu1}")
print(f"Q(X) = {comu2}")

manhatten = math.sqrt(sum([abs(a-b) for a,b in zip(hist1, hist2)]))
manhattenSum = " + ".join([f"|({a}-{b})|" for a,b in zip(hist1, hist2)])
print(f"manhatten: sqrt({manhattenSum}) = {manhatten}")

# p(X) = [0.3, 0.1, 0.2, 0, 0.4, 0]
# q(X) = [0, 0.1, 0.3, 0, 0.2, 0.4]
# E[p(X)] = 0.3*0 + 0.1*1 + 0.2*2 + 0*3 + 0.4*4 + 0*5 = 2.1
# E[q(X)] = 0*0 + 0.1*1 + 0.3*2 + 0*3 + 0.2*4 + 0.4*5 = 3.5
# P(X) = [0.3, 0.4, 0.6, 0.6, 1.0, 1.0]
# Q(X) = [0.0, 0.1, 0.4, 0.4, 0.6, 1.0]
# manhatten: sqrt(|(0.3-0)| + |(0.1-0.1)| + |(0.2-0.3)| + |(0-0)| + |(0.4-0.2)| + |(0-0.4)|) = 1.0