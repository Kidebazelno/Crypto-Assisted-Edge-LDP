import networkx as nx
import numpy as np
from math import comb
from scipy.sparse import csr_matrix

from tqdm import tqdm
import numba
import time



@numba.jit(cache=True)
def nCR(n, r):
    if n < r:
        return -1
    if n == r or r == 0:
        return 1
    max_val = max(r, n - r)
    min_val = min(r, n - r)
    return Multiplier(max_val + 1, n) / Multiplier(1, min_val)
@numba.jit(cache=True,parallel=True)
def kstar(k,graph):
    tmp = graph.sum(0)
    ans = 0
    for t in numba.prange(len(tmp)):
        i = tmp[t]
        ans += nCR(i,k)
    return ans

@numba.jit(cache=True,parallel=True)
def Multiplier(start, end):
    if start == end:
        return start
    res = 1

    for i in numba.prange(start,end+1):
        res *= i
    return res

@numba.jit(cache=True)
def graph_projection(d_max,graph:csr_matrix):
    nonzero = graph.nonzero()
    # all_degree matrix stores the degree of each node initially set to 0
    all_degree = np.zeros(graph.shape[0])
    # construct the projection graph
    
    for i,j in zip(nonzero[0],nonzero[1]):
        
        if i >= j:
            continue
        graph[i,j] = 0
        graph[j,i] = 0
        
        if all_degree[i] < d_max and all_degree[j] < d_max:
            graph[i,j] = 1
            graph[j,i] = 1
            
            all_degree[i] += 1
            all_degree[j] += 1
            
    
        
    
@numba.jit(cache=True)
def localLapk(d_max,graph:csr_matrix,k,epsilon=1.0,mode=0):
    delta = nCR(d_max,k-1)
    graph = graph.copy()
    node = 0
    ans = 0
    if mode == 1:
        graph_projection(d_max,graph)
    while node < graph.shape[0]:
        # get the neighbors of the node
        nonzero = graph.nonzero()

        # mode 0: random discard
        if mode == 0:
            # random discard until the number of neighbors is less than d_max
            neighbors = nonzero[1][nonzero[0] == node]
            if len(neighbors) > d_max:
                r = nCR(d_max,k)
            else:
                r = nCR(len(neighbors),k)
        # mode 1: graph projection
        elif mode == 1:
            neighbors = nonzero[1][nonzero[0] == node]
            r = nCR(len(neighbors),k)
        # add laplace noise
        r += np.random.laplace(0,delta/epsilon)
        # release the r
        ans += r

        node += 1
    return round(ans)

@numba.jit(cache=True)
def ope(a,b,n,node,message,seed=0):
    # order preserving encryption
    n = n+1
    np.random.seed(seed)
    mask = 0
    for i in range(n):
        if i < node:
            mask -= np.random.randint(0,256)
        elif i > node:
            mask += np.random.randint(0,256)
    noise = np.random.randint(0,256)

    return a*message + b + mask + noise

@numba.jit
def get_ope_key(n):
    # get the key for order preserving encryption
    # return a,b
    
    a = np.random.randint(1<<8,1<<10)
    b = np.random.randint(1,1<<8)
    return a,b


# def param_helper(mode,k,degree,a,b,n,i):
#     if mode == 1:
#         message = degree[i] - d_max if degree[i] > d_max else 0
#     elif mode == 2:
#         # print("degree:",degree[i],"d_max:",d_max)
#         if degree[i]>d_max:
#             message = l2_error(comb(degree[i],k),comb(d_max,k))
#             debug += message
#         else:
#             message = 0
#     # encrypt the message
#     message = ope(a,b,n,i,message)



@numba.jit(cache=True,parallel=True)
def param_selection(graph:csr_matrix,k=2,mode = 0,epsilon = 1.0):
    if mode == 0:
        d_max = 0
        # get the maximum degree
        graph = graph.sum(0)
        for tmp in numba.prange(len(graph)):
            # publish i with laplace noise
            i = graph[tmp]
            i += np.random.laplace(0,1/epsilon)
            i = round(i)
            # server get the maximum degree
            d_max = max(d_max,i)
        return d_max
    elif mode >= 1:
        # get the degree of each node, calculate the mean of the degree by homomorphic encryption
        n = graph.shape[0]
        # generate the public and private key
        a,b = get_ope_key(n) # order preserving encryption key, hide from the server
        lowest_loss = -1
        best_dmax = 0
        for d_max in (range(1,n)):
            loss = 0
            debug = 0
            g2 = graph.copy()
            degree = g2.sum(0)
            graph_projection(d_max,g2)
            for i in numba.prange(n):
                if mode == 1:
                    message = degree[i] - d_max if degree[i] > d_max else 0
                elif mode == 2:
                    # print("degree:",degree[i],"d_max:",d_max)
                    if degree[i]>d_max:
                        message = (nCR(degree[i],k)-nCR(d_max,k))**2
                        debug += message
                    else:
                        message = 0
                # encrypt the message
                message = ope(a,b,n,i,message)
                # send the message to the server
                loss += message
            # a client compute utility loss = n * variance = n*(delta/epsilon)^2
            delta = nCR(d_max,k-1)
            variance = n*(delta/epsilon)**2
            # print("d_max:",d_max,"variance:",variance,"debug:",debug)
            loss += ope(a,b,n,n,variance)
            if loss < lowest_loss or lowest_loss == -1:
                lowest_loss = loss
                best_dmax = d_max
        return best_dmax

@numba.jit
def l2_error(a,b):
    return (a-b)**2
@numba.jit
def real_dataset(A:csr_matrix,true_res=None):
    l2_errors = np.zeros(4)
    param_by_largest = np.zeros(4)
    direct_projection = np.zeros(4)
    param_large_direct_projection = np.zeros(4)
    epsilons = [0.5,1.0,1.5,2.0]
    print("Facebook dataset start")
    for i in range(len(epsilons)):
        epsilon = epsilons[i]
        print("epsilon:",epsilon)
        d_max = param_selection(A,2,mode=2,epsilon=epsilon)
        trial = 20

        tmp = np.zeros(trial)
        tmp1 = np.zeros(trial)
        tmp2 = np.zeros(trial)
        tmp3 = np.zeros(trial)
        
        for j in (range(10)):
            # print("CryptoAssisted d_max:",d_max)
            with numba.objmode(time1='f8'):
                time1 = time.perf_counter()
            print(j)
            priv_res = localLapk(d_max,A,2,mode=1,epsilon=epsilon)
            tmp[j] = (l2_error(priv_res,true_res))
            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon)
            tmp1[j] = (l2_error(priv_res,true_res))
            d_max = param_selection(A,2,mode=0,epsilon=epsilon/2)
            priv_res = localLapk(d_max,A,2,mode=1,epsilon=epsilon/2)
            tmp2[j]=(l2_error(priv_res,true_res))
            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon/2)
            tmp3[j]=(l2_error(priv_res,true_res))
            with numba.objmode():
                print('time: {}'.format(time.perf_counter() - time1))
        # f = open("facebook_dataset.txt","a")
        # f.writelines("epsilon:",epsilon)
        # f.writelines("l2 error:",tmp)
        # f.writelines("direct projection:",tmp1)
        # f.writelines("param by largest:",tmp2)
        # f.writelines("param large direct projection:",tmp3)
        # f.close()
        l2_errors[i] =(tmp.mean())
        direct_projection[i] =(tmp1.mean())
        param_by_largest[i] =(tmp2.mean())
        param_large_direct_projection[i] =(tmp3.mean())
    return l2_errors,direct_projection,param_by_largest,param_large_direct_projection

from matplotlib import pyplot as plt

setting = input("Enter 1 for epsilon vs l2 error, 2 for n vs l2 error, 3 for real dataset:")

if "1" in setting:
    epsilons = [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0]
    l2_errors = []
    param_by_largest = []
    direct_projection = []
    param_large_direct_projection = []
    for i in tqdm(range(55)):
        # generate a random graph
        n = 1<<5
        p = 0.33
        g = nx.random_graphs.gnp_random_graph(n, p)
        # get adjacency matrix
        A:csr_matrix = nx.linalg.graphmatrix.adjacency_matrix(g).astype(np.int32)
        A = A.todense()
        true_res = kstar(2,A)
        
        tmp = []
        tmp2 = []
        for epsilon in epsilons:
            d_max = param_selection(A,2,mode=2,epsilon=epsilon)
            # print("CryptoAssisted d_max:",d_max)
            # mode 1 of param_selection is the same
            priv_res = localLapk(d_max,A,2,mode=1,epsilon=epsilon)
            tmp.append(l2_error(priv_res,true_res))

            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon)
            tmp2.append(l2_error(priv_res,true_res))
        l2_errors.append(tmp)
        direct_projection.append(tmp2)

        tmp = []
        tmp2 = []
        for epsilon in epsilons:
            d_max = param_selection(A,2,mode=0,epsilon=epsilon/2)
            # print("ParamByLargest d_max:",d_max,", epsilon:",epsilon/2)
            priv_res = localLapk(d_max,A,2,mode=1,epsilon=epsilon/2)
            tmp.append(l2_error(priv_res,true_res))
            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon/2)
            tmp2.append(l2_error(priv_res,true_res))

        param_by_largest.append(tmp)
        param_large_direct_projection.append(tmp2)

        # garbage collection
        del g
        del A


    # get the average l2 error
    l2_errors = np.array(l2_errors)
    l2_errors = l2_errors.mean(0)
    direct_projection = np.array(direct_projection).mean(0)
    param_by_largest = np.array(param_by_largest).mean(0)
    param_large_direct_projection = np.array(param_large_direct_projection).mean(0)



    plt.plot(epsilons,param_large_direct_projection,'go-',label="ParamByLargest + DirectDeletion")
    plt.plot(epsilons,direct_projection,'ys-',label="CryptoAssisted + DirectDeletion")
    plt.plot(epsilons,param_by_largest,'bo-',label="param by largest + addProjection")
    plt.plot(epsilons,l2_errors,'ro-',label="CryptoAssisted + addProjection")

    plt.xlabel("epsilon")
    plt.ylabel("l2 error")
    plt.legend()
    # save the figure
    plt.savefig("epsilon_vs_l2_error.png")

if "2" in setting:
    # Now compare n vs l2 error
    l2_errors = []
    param_by_largest = []
    direct_projection = []
    param_large_direct_projection = []
    bit = 11
    for i in (range(1,bit)):
        print("n:",1<<i)
        tmp = []
        tmp2 = []
        tmp3 = []
        tmp1 = []
        for j in tqdm(range(13)):
            # generate a random graph
            n = 1<<i
            p = 0.3
            g = nx.random_graphs.gnp_random_graph(n, p)
            # get adjacency matrix
            A = nx.linalg.graphmatrix.adjacency_matrix(g).astype(np.int32).todense()
            d_max = param_selection(A,2,mode=2)
            epsilon = 1.0
            priv_res = localLapk(d_max,A,2,mode=1,epsilon=epsilon)
            true_res = kstar(2,A)
            tmp.append(l2_error(priv_res,true_res))

            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon)
            tmp1.append(l2_error(priv_res,true_res))

            d_max = param_selection(A,2,mode=0,epsilon=epsilon/2)
            priv_res = localLapk(d_max,A,2,mode=1,epsilon=epsilon/2)
            tmp2.append(l2_error(priv_res,true_res))

            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon/2)
            tmp3.append(l2_error(priv_res,true_res))
            # garbage collection
            del g
            del A

        l2_errors.append(np.array(tmp).mean())
        direct_projection.append(np.array(tmp1).mean())
        param_by_largest.append(np.array(tmp2).mean())
        param_large_direct_projection.append(np.array(tmp3).mean())
        print("l2 error:",l2_errors)
        print("direct projection:",direct_projection)
        print("param by largest:",param_by_largest)
        print("param large direct projection:",param_large_direct_projection)
    f = open("n_vs_l2_error.txt","w")
    print("n vs l2 error",file=f)
    print("l2 error:",l2_errors,file=f)
    print("direct projection:",direct_projection,file=f)
    print("param by largest:",param_by_largest,file=f)
    print("param large direct projection:",param_large_direct_projection,file=f)
    f.close()

    plt.clf()
    plt.plot([1<<i for i in range(1,bit)],param_large_direct_projection,'go-',label="ParamByLargest + DirectDeletion")
    plt.plot([1<<i for i in range(1,bit)],direct_projection,'ys-',label="CryptoAssisted + DirectDeletion")
    plt.plot([1<<i for i in range(1,bit)],param_by_largest,'bo-',label="param by largest + addProjection")
    plt.plot([1<<i for i in range(1,bit)],l2_errors,'ro-',label="CryptoAssisted + addProjection")

    plt.xlabel("n")
    plt.ylabel("l2 error")
    plt.legend()
    # save the figure
    plt.savefig("n_vs_l2_error.png")


if "3" in setting:

    # load the facebook dataset
    G = nx.read_edgelist("facebook_combined.txt",create_using=nx.Graph(),nodetype=int)
    
    A = nx.linalg.graphmatrix.adjacency_matrix(G).astype(np.int32).todense()
    # get the true result
    true_res = kstar(2,A)
    epsilons = [0.5,1.0,1.5,2.0]
    
    l2_errors,direct_projection,param_by_largest,param_large_direct_projection = real_dataset(A,true_res)
    print("l2 error:",l2_errors)
    print("direct projection:",direct_projection)
    print("param by largest:",param_by_largest)
    print("param large direct projection:",param_large_direct_projection)
    plt.clf()
    plt.plot(epsilons,param_large_direct_projection,'go-',label="ParamByLargest + DirectDeletion")
    plt.plot(epsilons,direct_projection,'ys-',label="CryptoAssisted + DirectDeletion")
    plt.plot(epsilons,param_by_largest,'bo-',label="param by largest + addProjection")
    plt.plot(epsilons,l2_errors,'ro-',label="CryptoAssisted + addProjection")

    plt.xlabel("epsilon")
    plt.ylabel("l2 error")
    plt.legend()
    plt.title("Facebook dataset")
    # save the figure
    plt.savefig("facebook_dataset.png")

if "4" in setting:
    # compare crypto assisted mode 1 and mode 2
    l2_errors = []
    projection2 = []
    epsilons = [0.5,1.0,1.5,2.0]
    for i in tqdm(range(10)):
        # generate a random graph
        n = 1<<8
        p = 0.3
        g = nx.random_graphs.gnp_random_graph(n, p)
        # get adjacency matrix
        A:csr_matrix = nx.linalg.graphmatrix.adjacency_matrix(g).astype(np.int32)
        A = A.todense()
        true_res = kstar(2,A)
        
        tmp = []
        tmp2 = []
        for epsilon in epsilons:
            d_max = param_selection(A,2,mode=2,epsilon=epsilon)
            # print("CryptoAssisted d_max:",d_max)
            # mode 1 of param_selection is the same
            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon)
            tmp.append(l2_error(priv_res,true_res))

            d_max = param_selection(A,2,mode=1,epsilon=epsilon)
            priv_res = localLapk(d_max,A,2,mode=0,epsilon=epsilon)
            tmp2.append(l2_error(priv_res,true_res))
            
        l2_errors.append(tmp)
        projection2.append(tmp2)
        # garbage collection
        del g
        del A

    # get the average l2 error
    l2_errors = np.array(l2_errors).mean(0)
    projection2 = np.array(projection2).mean(0)

    plt.plot(epsilons,projection2,'go-',label="CryptoAssisted mode 1")
    plt.plot(epsilons,l2_errors,'ro-',label="CryptoAssisted mode 2")

    plt.xlabel("epsilon")
    plt.ylabel("l2 error")
    plt.legend()
    # save the figure
    plt.savefig("CryptoAssisted_mode1_vs_mode2.png")
        
