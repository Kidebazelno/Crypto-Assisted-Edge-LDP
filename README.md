# Edge Local Differential Privacy for Graph Statistic
## Description
In this project, I modified the proposed method in [2] so it can be applied to the algorithm proposed by [1]. With the modification, I am able to get lower l2-loss given the same privacy budget.
## Usage
Run the command
```
python3 edge_ldp.py
```
The output will be the comparison of the modified method and the original method in [1].

## Results
![](https://github.com/Kidebazelno/Crypto-Assisted-Edge-LDP/blob/main/Test%20Results/facebook_dataset.png) ![](https://github.com/Kidebazelno/Crypto-Assisted-Edge-LDP/blob/main/Test%20Results/n_vs_l2_error.png)


## Reference
[1] J. Imola, T. Murakami, and K. Chaudhuri, “Locally differentially private analysis of graph statistics,” in 30th USENIX Security Symposium (USENIX Security 21), 2021, pp. 983–1000. \
[2] S. Liu, Y. Cao, T. Murakami and M. Yoshikawa, "A Crypto-Assisted Approach for Publishing Graph Statistics with Node Local Differential Privacy," 2022 IEEE International Conference on Big Data (Big Data), Osaka, Japan, 2022, pp. 5765-5774,
doi: 10.1109/BigData55660.2022.10020435
