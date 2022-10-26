import math

hist1 = [0.3,0.1,0.2,0,0.4,0]
hist2 = [0,0.1,0.3,0,0.2,0.4]

exp1 = sum([p*x for x,p in enumerate(hist1)])
print(f"E[p] = {exp1}")
exp2 = sum([p*x for x,p in enumerate(hist2)])
print(f"E[q] = {exp2}")

comu1 = [0.3,0.4,0.6,0.6,1.0,1.0]
comu2 = [0.0,0.1,0.4,0.0,0.6,1.0]

print(f"comu1: {comu1}")
print(f"comu2: {comu2}")

manhatten = math.sqrt(sum([(a-b)**2 for a,b in zip(hist1, hist2)]))
print(f"manhatten: {manhatten}")

# E[p] = 2.1
# E[q] = 3.5
# comu1: [0.3, 0.4, 0.6, 0.6, 1.0, 1.0]
# comu2: [0.0, 0.1, 0.4, 0.0, 0.6, 1.0]
# manhatten: 0.5477225575051662