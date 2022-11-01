# Gruppe 26
# Laszlo Korte
# Alexander Remmes-Weitz

import math

hist1 = [0.3,0.1,0.2,0,0.4,0]
hist2 = [0,0.1,0.3,0,0.2,0.4]

exp1 = sum([p*x for x,p in enumerate(hist1)])
sum1 = " + ".join([f"{p}*{x}" for x,p in enumerate(hist1)])
print(f"E[p] = {sum1} = {exp1}")
exp2 = sum([p*x for x,p in enumerate(hist2)])
sum2 = " + ".join([f"{p}*{x}" for x,p in enumerate(hist2)])
print(f"E[q] = {sum2} = {exp2}")

comu1 = [0.3,0.4,0.6,0.6,1.0,1.0]
comu2 = [0.0,0.1,0.4,0.4,0.6,1.0]

print(f"comu1: {comu1}")
print(f"comu2: {comu2}")

manhatten = math.sqrt(sum([abs(a-b) for a,b in zip(hist1, hist2)]))
manhattenSum = " + ".join([f"|({a}-{b})|" for a,b in zip(hist1, hist2)])
print(f"manhatten: sqrt({manhattenSum}) = {manhatten}")

# E[p] = 0.3*0 + 0.1*1 + 0.2*2 + 0*3 + 0.4*4 + 0*5 = 2.1
# E[q] = 0*0 + 0.1*1 + 0.3*2 + 0*3 + 0.2*4 + 0.4*5 = 3.5
# comu1: [0.3, 0.4, 0.6, 0.6, 1.0, 1.0]
# comu2: [0.0, 0.1, 0.4, 0.4, 0.6, 1.0]
# manhatten: sqrt(|(0.3-0)| + |(0.1-0.1)| + |(0.2-0.3)| + |(0-0)| + |(0.4-0.2)| + |(0-0.4)|) = 1.0