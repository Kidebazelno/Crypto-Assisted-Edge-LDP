import numpy as np

l2_errors = []
param_by_largest = []
direct_projection = []
param_large_direct_projection = []
epsilons = [0.5,1.0,1.5,2.0]

# read data
f = open("facebook_dataset_post.txt", "r")
data = f.readlines()
f.close()

i = 0
epsilon = [[[]]*4]*len(epsilons)
print(epsilon)
while i < len(data):
    now_epsilon = float(data[i].split(": ")[1])
    tmp = eval(data[i+1].split("=")[1])
    tmp1 = eval(data[i+2].split("=")[1])
    tmp2 = eval(data[i+3].split("=")[1])
    tmp3 = eval(data[i+4].split("=")[1])
    print(epsilons.index(now_epsilon),now_epsilon)
    print(epsilon[epsilons.index(now_epsilon)][0])
    print(tmp)
    epsilon[epsilons.index(now_epsilon)][0]+=(tmp)
    epsilon[epsilons.index(now_epsilon)][1]+=(tmp1)
    epsilon[epsilons.index(now_epsilon)][2]+=(tmp2)
    epsilon[epsilons.index(now_epsilon)][3]+=(tmp3)
    i += 5


# get the result

for i in range(len(epsilons)):
    print(epsilon[i])
    l2_errors.append(np.array(epsilon[i])[:,0].mean())
    direct_projection.append(np.array(epsilon[i])[:,1].mean())
    param_by_largest.append(np.array(epsilon[i])[:,2].mean())
    param_large_direct_projection.append(np.array(epsilon[i])[:,3].mean())

# l2_errors.append(np.array(tmp).mean())
# direct_projection.append(np.array(tmp1).mean())
# param_by_largest.append(np.array(tmp2).mean())
# param_large_direct_projection.append(np.array(tmp3).mean())

# plot
import matplotlib.pyplot as plt

plt.plot(epsilons, l2_errors, label="l2 error")
plt.plot(epsilons, direct_projection, label="direct projection")
plt.plot(epsilons, param_by_largest, label="param by largest")
plt.plot(epsilons, param_large_direct_projection, label="param large direct projection")
plt.legend()
plt.xlabel("epsilon")
plt.ylabel("error")
plt.title("Facebook dataset")
plt.savefig("facebook_dataset.png")