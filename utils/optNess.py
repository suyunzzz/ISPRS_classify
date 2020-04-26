'''
基于最小化香农熵，为点云中的每一个点寻找最优的邻域点的个数
1.读取点云，knn计算每一个点的邻域点
2.为最优秀邻域大小创建空间，创建香农熵矩阵
3.计算香农熵，遍历每一个点，对每一个点再遍历不同的邻域k值
4.选取上述香农熵矩阵的最小的香农熵对应的k值，作为最优的k值
'''


import  numpy as np
import sys
import os
from sklearn.neighbors import NearestNeighbors

# 计算最优近邻点 返回一个矩阵
def getOptNess(pt_in,k_min,k_max,delta_k):
    point_ID_max=pt_in.shape[0]
    k=list(range(k_min,k_max+1,delta_k))
    print('k:{}'.format(k))
    # 获取近邻点索引
    data_pts=pt_in[:,0:3]
    k_plus_1=max(k)+1
    # print('k_plus_1:{}'.format(k_plus_1))

    num_k=len(k)  #获取邻域的尺度数目  这里是8  一共计算八个尺度


    neigh = NearestNeighbors(n_neighbors=k_plus_1)   #最近的20个点
    neigh.fit(data_pts)
    index = neigh.kneighbors(data_pts,return_distance=False) #z最近邻的点的索引
    # print('index:\n{}'.format(index))
    print('index.shape:{}'.format(index.shape))  #index是一个索引矩阵 n*(k_max+1)

    # 初始化香农熵矩阵
    Shannon_entropy = np.zeros((point_ID_max,num_k),float)   # n*8
    opt_nn_size=np.zeros((point_ID_max,1),int)  #最终返回值 代表每一个点的最优k值


    #计算香农熵
    for i in range(point_ID_max):  #遍历每一个点
        Shannon_entropy_real=np.zeros((1,num_k),float)

        for j in range(num_k):  #遍历每一个尺度
            # print('index[i,0:k[j]+1]:{}'.format(index[i,0:k[j]+1]))   #一个list []
            P=data_pts[index[i,0:k[j]+1],:]

            #计算有几个点
            m=P.shape[0]

            #计算这个P的协方差矩阵
            pp=P
            pp[:, 0] = pp[:, 0] - np.mean(pp[:, 0])
            pp[:, 1] = pp[:, 1] - np.mean(pp[:, 1])
            pp[:, 2] = pp[:, 2] - np.mean(pp[:, 2])
            a = pp.T  # 矩阵转置
            b = np.matmul(pp.T, pp)  # 协方差矩阵
            C=b/(m-1)

            #计算特征值
            fev = np.linalg.eigvals(C)  # 特征值
            fev = (np.sort(fev))  # 特征值排序，从小到大
            # print('fev:{}'.format(fev))

            epsilon_to_add=1e-8

            if fev[0]<=0:
                fev[0]=epsilon_to_add
                if fev[1]<=0:
                    fev[1]=epsilon_to_add
                    if fev[2]<=0:
                        fev[2]=epsilon_to_add

            # print('fev:{}'.format(fev))
            # 归一化特征值
            fev=fev/np.sum(fev)
            # print('normalized fev:{}'.format(fev))

            #计算香农熵
            Shannon_entropy_cal=-( fev[0]*np.log(fev[0])+ fev[1]*np.log(fev[1])+ fev[2]*np.log(fev[2]) )
            # print('Shannon_entropy_cal:{}'.format(Shannon_entropy_cal))
            # Shannon_entropy_real[j]=np.real(Shannon_entropy_cal)
            Shannon_entropy_real[0,j] =Shannon_entropy_cal

        Shannon_entropy[i,:]=Shannon_entropy_real

        # 选择最小的香农熵
        min_entry_of_Shannon_entropy=np.argmin(Shannon_entropy_real)
        # print("min_entry_of_Shannon_entropy:{}".format(min_entry_of_Shannon_entropy))
        opt_nn_size[i,0]=k[min_entry_of_Shannon_entropy]
    return opt_nn_size



if __name__ == '__main__':
    pt_in=np.loadtxt('./test/buildings_and_trees.txt')
    optNN=getOptNess(pt_in,2,10,1)
    print('optNN.shape:{}'.format(optNN.shape))
    print('optNN:\n{}'.format(optNN))
    np.savetxt('./optNess.txt',optNN)
