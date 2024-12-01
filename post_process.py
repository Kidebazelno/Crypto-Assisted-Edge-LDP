# combine the result in output data

f = open("facebook_dataset.txt", "r")
data = f.readlines()
f.close()

"""epsilon: 2.0
l2 error: [6441988644, 5729278864, 1692005956, 1398909604, 116380944]
direct projection: [4344391744, 76615009, 479697604, 1002102336, 1277419081]
param by largest: [12442287025, 4166186116, 15850306404, 728892004, 21414517569]
param large direct projection: [27876977296, 4338461689, 2200829569, 11390625, 123921424]
epsilon: 0.5
l2 error: [108300886281.0, 92442753936.0, 26352976896.0, 23083940356.0, 1784217600.0, 29529329281.0]
direct projection: [74001953089.0, 1349166361.0, 7834020100.0, 16611600996.0, 20068688896.0, 656538129.0]
param by largest: [199970846761.0, 65297114089.0, 250641410881.0, 11413289889.0, 332699393601.0, 20948509696.0]
param large direct projection: [441947073681.0, 69444371529.0, 36045300736.0, 201668401.0, 1938640900.0, 6566833296.0]
"""

i = 0
epsilon = {}
while i < len(data):
    now_epsilon = float(data[i].split(": ")[1])
    l2_error = eval(data[i+1].split(": ")[1])
    direct_projection = eval(data[i+2].split(": ")[1])
    param_by_largest = eval(data[i+3].split(": ")[1])
    param_large_direct_projection = eval(data[i+4].split(": ")[1])
    if now_epsilon not in epsilon:
        epsilon[now_epsilon] = {}
        epsilon[now_epsilon]["l2_error"] = l2_error
        epsilon[now_epsilon]["direct_projection"] = direct_projection
        epsilon[now_epsilon]["param_by_largest"] = param_by_largest
        epsilon[now_epsilon]["param_large_direct_projection"] = param_large_direct_projection
    else:
        epsilon[now_epsilon]["l2_error"].extend(l2_error)
        epsilon[now_epsilon]["direct_projection"].extend(direct_projection)
        epsilon[now_epsilon]["param_by_largest"].extend(param_by_largest)
        epsilon[now_epsilon]["param_large_direct_projection"].extend(param_large_direct_projection)
    i += 5

# write back to file
f = open("facebook_dataset_post.txt", "w")
for key in epsilon:
    f.write("epsilon: " + str(key) + "\n")
    f.write("l2 error= " + str(epsilon[key]["l2_error"]) + "\n")
    f.write("direct projection= " + str(epsilon[key]["direct_projection"]) + "\n")
    f.write("param by largest= " + str(epsilon[key]["param_by_largest"]) + "\n")
    f.write("param large direct projection= " + str(epsilon[key]["param_large_direct_projection"]) + "\n")
f.close()