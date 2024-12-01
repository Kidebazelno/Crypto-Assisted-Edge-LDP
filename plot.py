l2_errors= [4.176470588235294, 6.117647058823529, 207.05882352941177, 2728.294117647059, 11142.176470588236, 117190.29411764706, 706404.7647058824, 4584529.823529412, 28188859.529411763, 171805842.6470588, 2187985976.4117646]
direct_projection = [4.235294117647059, 13.882352941176471, 384.2352941176471, 2943.0, 19476.647058823528, 138612.29411764705, 392771.4117647059, 7699431.294117647, 23435727.352941178, 266366658.94117647, 2407649773.529412]
param_by_largest = [187.41176470588235, 438.0, 1501.8823529411766, 22184.470588235294, 76364.11764705883, 596084.8235294118, 3443278.117647059, 27243811.647058822, 135609044.5882353, 1092807574.4117646, 7747860513.176471]
param_large_direct_projection = [161.11764705882354, 591.1764705882352, 1838.4117647058824, 6713.058823529412, 45934.35294117647, 399670.9411764706, 3248447.3529411764, 19423986.647058822, 165148726.7647059, 1037635240.5294118, 8109969106.529411]

import matplotlib.pyplot as plt
import numpy as np

bit = len(l2_errors)+1
# data
plt.plot([1<<i for i in range(1,bit)],param_large_direct_projection,'go-',label="ParamByLargest + DirectDeletion")
plt.plot([1<<i for i in range(1,bit)],direct_projection,'ys-',label="CryptoAssisted + DirectDeletion")
plt.plot([1<<i for i in range(1,bit)],param_by_largest,'bo-',label="param by largest + addProjection")
plt.plot([1<<i for i in range(1,bit)],l2_errors,'ro-',label="CryptoAssisted + addProjection")

plt.xlabel("n")
plt.ylabel("l2 error")
plt.legend()
# save the figure
plt.savefig("n_vs_l2_error.png")