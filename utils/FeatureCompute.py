'''
计算输入点云的每一个点的特征向量
'''

#1.读入点云
#2 为每一个点计算k邻域
#3 计算邻域内的协方差矩阵
#4 计算曲率、粗糙度、发散状指数、面状指数、法向量与z轴夹角、邻域内高度差（邻域特征）
#5 保存为一个n*F的矩阵 ，n为点数，f为特征数

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os

from data_noGround.Scaled.data_split import splitpointcloud
from utils import optNess

#计算邻域内的点距离拟合平面的距离
def CaculateAverageSquareDistance(pointMat):  # 输入 ppointMat为一个矩阵，返回距离的平方和
    num = pointMat.shape[0]
    B = np.zeros((pointMat.shape[0],3))
    one = np.ones((pointMat.shape[0],1))
    B[:,0] = pointMat[:,0]
    B[:,1] = pointMat[:,1]
    B[:,2] = one[:,0]
    l = pointMat[:,2]
    BTB = np.matmul(B.T,B)
    BTB_1 = np.linalg.pinv(BTB)
    temp = np.matmul(BTB_1,B.T)
    result = np.matmul(temp,l)
    V  = np.matmul(B,result)-l
    sum = 0
    for i in range (0,V.shape[0]):
        sum = sum+V[i]**2
    return sum/V.shape[0]


# K=17 #最近邻域的点的数目
#计算粗糙度,输入为point点，pointMat矩阵，K对该点的最近邻数目
def CaculateRoughness(point,pointMat,K):  #输入point 应该是一个点,pointMat是这个点云,返回一个值，代表该点邻域的粗糙度

    # 计算K个近邻点
    neigh = NearestNeighbors(n_neighbors=K)   #最近的20个点
    neigh.fit(pointMat)
    index = neigh.kneighbors([point],return_distance=False) #z最近邻的点的索引


    avedis2 = CaculateAverageSquareDistance(pointMat[index].reshape(K,3))   #每一个点邻域的表面粗超度


    return avedis2*10000


# 邻域分析计算发散状指数
#输入：point：点,pointMat：点云，K：该点的最近邻数目；输出：该点的发散状指数
# 计算邻域点的高程方差
# 输入：点云矩阵n*4 x y z label;K近邻大小
# 输出：高程方差矩阵 n*1 代表每一个点的方差大小
# 输出：cur每一个点的曲率
def GetNeiAnaAndElevationVariance(point,pointMat,K):
    p1 = pointMat[:, 0:3]
    neigh = NearestNeighbors(n_neighbors=K)
    neigh.fit(pointMat)
    index = neigh.kneighbors([point], return_distance=False)

    pp = p1[index].reshape(K, 3)  #指定点的邻域点云pp

    # print('邻域点-pp:{}'.format(pp.shape))


    pp[:, 0] = pp[:, 0] - np.mean(pp[:, 0])
    pp[:, 1] = pp[:, 1] - np.mean(pp[:, 1])
    pp[:, 2] = pp[:, 2] - np.mean(pp[:, 2])  # z值的差
    ppNeighbor = pp[:,2]  # 用于计算高程方差
    # print('高程去中心化-pp[:,2]:{}'.format(pp[:,2].shape))
    a = pp.T  # 矩阵转置
    b = np.matmul(pp.T, pp)  # 协方差矩阵
    fev = np.linalg.eigvals(b)  # 特征值
    fev = (np.sort(fev))  # 特征值排序，从小到大
    scattering= (fev[0]) / fev[2] * 1000  # λ3/λ1 (λ3<λ2<λ1)  计算发散指数

    #计算曲率,最小特征值占得比重
    cur=fev[0]/sum(fev)  # λ3/λ1+λ2+λ3



    # 计算高程方差
    # print('ppNeighbor shape:{}'.format(ppNeighbor.shape))   # 应该是一个矩阵
    # print('ppNeighbor:{} '.format(ppNeighbor))
    ElevationVariance=ppNeighbor*2

    ElevationVariance=np.sum(ElevationVariance)/ppNeighbor.shape   # 计算平均值
    # print('ElevationVariance shape:{}'.format(ElevationVariance.shape))
    # print('ElevationVariance:{}'.format(ElevationVariance))

    return scattering,ElevationVariance,cur

# 计算邻域点的高程方差
# 输入：点云矩阵n*4 x y z label;K近邻大小
# 输出：高程方差矩阵 n*1 代表每一个点的方差大小



# 组成特征向量【发散状指数，粗糙度】
# 输入：点云p（n*3）,最优的最近邻数目optK （n*1的矩阵）
# 输出：nxm的特征矩阵  ，m为特征的个数
def GetFeatureVector(p,optN):
    print('optN shape:{}'.format(optN.shape))
    # fv=np.zeros((p.shape[0],4),float)  # 预先开辟的空间，存放特征向量
    fv=np.zeros((p.shape[0],2),float)  # 预先开辟的空间，存放特征向量
    for i in range(p.shape[0]):   #遍历每一个点
        print('--------------------------------------')
        fv[i, 0],_,_ = GetNeiAnaAndElevationVariance(p[i],p,optN[i,0]) #scattering,ElevationVariance,cur
        fv[i, 1] = CaculateRoughness(p[i],p,optN[i,0]) #粗糙度

        print('完成度：{}%'.format((i/p.shape[0])*100))
        i+=1
        print('----------------------------------------')
    return fv


#使用规定的K近邻来计算特征向量
def GetFeatureVector_K(p,K=10):
    print('K size:{}'.format(K))
    # fv=np.zeros((p.shape[0],4),float)  # 预先开辟的空间，存放特征向量
    fv=np.zeros((p.shape[0],2),float)  # 预先开辟的空间，存放特征向量
    for i in range(p.shape[0]):   #遍历每一个点
        print('--------------------------------------')
        fv[i, 0],_,_ = GetNeiAnaAndElevationVariance(p[i],p,K) #scattering,ElevationVariance,cur
        fv[i, 1] = CaculateRoughness(p[i],p,K) #粗糙度

        print('完成度：{}%'.format((i/p.shape[0])*100))
        i+=1
        print('----------------------------------------')
    return fv

#2020-3-7更新
#数据标准化 将不同大小的特征归一化到0~1，更容易处理
from sklearn import preprocessing
def FeatureNormalize(Feature_in):
    Fecture_out=np.zeros((Feature_in.shape[0],Feature_in.shape[1]),dtype=np.float64)
    Feature_out=preprocessing.scale(Feature_in)
    return Feature_out

# if __name__ == '__main__':
#
#     pt=np.loadtxt('./../data_noGround/Scaled/Scaled_NoGround_All1-5.txt')
#     print('pt:{}'.format(pt.shape))
#     buildings,trees=splitpointcloud(pt)  #调用函数获取分类后的点云
#     print('buildings:{},trees:{}'.format(buildings.shape,trees.shape))  #buildings:(12578, 4),trees:(39110, 4)
#     print('输入点云的格式：',buildings[:,0:3].shape)
#
#     # 计算buildings的特征向量
#     optN = optNess.getOptNess(buildings[:,0:3].reshape(-1,3), 5, 10, 1)
#     fv_buildings=GetFeatureVector(buildings[:,0:3].reshape(-1,3),optN) #计算特征向量
#     print('buildings特征向量格式：',fv_buildings.shape)
#     # print('fv:\n',fv)
#
#
#     # 计算trees的特征向量
#     optN = optNess.getOptNess(trees[:,0:3].reshape(-1,3), 5, 10, 1)
#     fv_trees=GetFeatureVector(trees[:,0:3].reshape(-1,3),optN) #计算特征向量
#     print('trees特征向量格式：',fv_trees.shape)
#
#
#     #保存Fv
#     type=-1 #buildings的类型设置为-1
#     if os.path.exists('./../Features/OptN_AllFeatures.txt'):
#         os.remove('./../Features/OptN_AllFeatures.txt')
#
#     f = open('./../Features/OptN_AllFeatures.txt', 'a')  #以append的格式打开文件
#     for i in range(fv_buildings.shape[0]):
#
#         f.write(str(fv_buildings[i,0])+","+str(fv_buildings[i,1])+","+str(fv_buildings[i,2])+","+
#                 str(fv_buildings[i,3])+","+str(type)+"\n")
#     f.close()
#
#     #保存Fv
#     type=1 #trees的类型设置为1
#     f = open('./../Features/OptN_AllFeatures.txt', 'a')
#     for i in range(fv_trees.shape[0]):
#
#         f.write(str(fv_trees[i,0])+","+str(fv_trees[i,1])+","+str(fv_trees[i,2])+","+
#                 str(fv_trees[i,3])+","+str(type)+"\n")
#     f.close()


# '''
# 计算两类的特征向量，生成一个文件
# '''
# if __name__ == '__main__':
#     #计算tree的特征
#     p = np.loadtxt("./../data_noGround/Scaled/trees.txt" )
#     pointIn=np.zeros((p.shape[0],3),float)
#     pointIn[:,:]=p[:,0:3]
#
#     print('输入点云的格式：',pointIn.shape)
#     optN = optNess.getOptNess(pointIn, 10, 20, 2)
#     fv=GetFeatureVector(pointIn,optN) #计算特征向量
#     print('特征向量格式：',fv.shape)
#     # fv_Normalized=FeatureNormalize(fv) #标准化后的特征向量
#     # print('fv_Normalized: ',fv_Normalized.shape)
#     # print(fv)
#     # print('-------------------------')
#     # print(fv_Normalized)
#
#
#     #保存Fv
#     type=-1 #树的类型设置为-1
#     if os.path.exists('./train/Features/OptN_AllFeatures.txt'):
#         os.remove('./train/Features/OptN_AllFeatures.txt')
#
#     f = open('./train/Features/OptN_AllFeatures.txt', 'a')  #以append的格式打开文件
#     for i in range(fv.shape[0]):
#
#         f.write(str(fv[i,0])+","+str(fv[i,1])+","+str(type)+"\n")
#     f.close()
#
#
# # -----------------------------------------------------------------------------------------
#
# # 计算building的特征
#     p_road = np.loadtxt("./train/buildings.txt" )
#     pointIn=np.zeros((p_road.shape[0],3),float)
#     pointIn[:,:]=p_road[:,0:3]
#
#     print(pointIn.shape)
#     optN=optNess.getOptNess(pointIn,10,20,2)
#     fv_road=GetFeatureVector(pointIn,optN) #得到特征向量
#     print(fv_road.shape)
#
#     # fv_road_Normalized = FeatureNormalize(fv_road)  # 标准化后的特征向量
#     # print(fv_road_Normalized)
#
#     #保存Fv
#     type=1 #building的类型设置为1
#
#     f = open('./train/Features/OptN_AllFeatures.txt', 'a')
#     for i in range(fv_road.shape[0]):
#
#         f.write(str(fv_road[i,0])+","+str(fv_road[i,1])+","+str(type)+"\n")
#     f.close()
#
#     #可视化特征
#     plt.scatter(fv[:, 0], fv[:, 1], c='g')#树红色 type 为1
#     plt.scatter(fv_road[:,0],fv_road[:,1],c='r') #buildings红色 type 为-1
#     plt.title('Features')
#     plt.show()