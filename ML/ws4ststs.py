import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

s = np.array([10, 40 ,20, 85, 105 ,85 ,270 ,40, 60 ,90, 360, 30 ,520 ,210, 45 ,16, 6 ,420, 90 ,515, 71, 448, 353, 526, 279,
279, 505 ,157 ,240 ,840, 240 ,180, 300, 450 ,225, 60, 300])
# m = np.mean(s)
# print(m)
# std = np.std(s)
# print(std)

#q = np.array([235 ,210 ,95 ,146, 195 ,840, 185 ,610 ,680 ,990 ,146 ,404, 119 ,47, 9, 4, 10, 169, 270, 95 ,329 ,151 ,211,
                    # 127 ,154, 35 ,225, 140 ,158, 116, 46 ,113 ,149 ,420 ,120 ,45, 10 ,18 ,105])
# print(len(q))

q = np.array([164, 272, 261, 248, 235, 192, 203, 278, 268, 230, 242, 305, 286, 310,345, 289, 326, 335, 297, 328, 400, 228, 194, 338, 252])
m = np.mean(q)
print("mean of q is ",m)
s = np.std(q)
print(s)

sdvc = []
for i in range (len(q)):
    sd = ((q[i] - m)**2 )
    sdvc.append(sd)
v = (np.sum(sdvc)) / len(q) - 1
print(sdvc)
print(np.sqrt(v))
print(v)








