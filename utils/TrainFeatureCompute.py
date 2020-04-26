import sys
sys.path.append('./..')
from utils.FeatureCompute import GetFeatureVector


from data_noGround.Scaled.data_split import splitpointcloud
from utils import optNess
import os

import numpy as np

if __name__ == '__main__':
    pt=np.loadtxt('./../data_noGround/NoGround_data5_cat.txt')
    print('pt:{}'.format(pt.shape))
    buildings,trees=splitpointcloud(pt)  #调用函数获取分类后的点云
    print('buildings:{},trees:{}'.format(buildings.shape,trees.shape))  #buildings:(12578, 4),trees:(39110, 4)
    print('输入点云的格式：',buildings[:,0:3].shape)

    # 计算buildings的特征向量
    optN = optNess.getOptNess(buildings[:,0:3].reshape(-1,3), 20, 30, 2)
    fv_buildings=GetFeatureVector(buildings[:,0:3].reshape(-1,3),optN) #计算特征向量
    print('buildings特征向量格式：',fv_buildings.shape)
    # print('fv:\n',fv)


    # 计算trees的特征向量
    optN = optNess.getOptNess(trees[:,0:3].reshape(-1,3), 5, 10, 1)
    fv_trees=GetFeatureVector(trees[:,0:3].reshape(-1,3),optN) #计算特征向量
    print('trees特征向量格式：',fv_trees.shape)


    #保存Fv
    type=-1 #buildings的类型设置为-1
    if os.path.exists('./../Features/OptN20-30_AllFeaturesTrain5.txt'):
        os.remove('./../Features/OptN20-30_AllFeaturesTrain5.txt')

    f = open('./../Features/OptN20-30_AllFeaturesTrain5.txt', 'a')  #以append的格式打开文件
    for i in range(fv_buildings.shape[0]):

        # f.write(str(fv_buildings[i,0])+","+str(fv_buildings[i,1])+","+str(fv_buildings[i,2])+","+
        #         str(fv_buildings[i,3])+","+str(type)+"\n")
        f.write(str(fv_buildings[i,0])+","+str(fv_buildings[i,1])+","+str(type)+"\n")
    f.close()

    #保存Fv Trees
    type=1 #trees的类型设置为1
    f = open('./../Features/OptN20-30_AllFeaturesTrain5.txt', 'a')
    for i in range(fv_trees.shape[0]):

        f.write(str(fv_trees[i,0])+","+str(fv_trees[i,1])+","+str(type)+"\n")
    f.close()

    # ******************************************************************************************************
