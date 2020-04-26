'''
选取出地面和树木，在其中选取一部分，用做测试集
'''
import sys
import os

# 读取目录下所有内容：当前路径下所有非目录子文件
# 返回文件名 []
def getallfiles(file_dir):
    for root,dirs,files in os.walk(file_dir):
        print('files:{}'.format(files))
        return  files

# 输入为一个txt文件，输出为新的提取好的文件
def data_cat(filename):

    f_in=open(filename+'.pts')
    p=open(filename+'_TreesAndRoof.pts','w')
    lines=f_in.readlines()
    # print(len(lines))
    i=1
    cat1=0
    cat2=0
    cat4=0

    for line in lines:
        line=line.split(' ')


        # print(line[0])

        if line[-1]=='5\n': #roof
            p.write(line[0]+' '+line[1]+' '+line[2]+" "+line[3]+" "+line[-1]+'\n')
            cat1+=1

        elif line[-1]=='8\n':  #trees
            p.write(line[0]+' '+line[1]+' '+line[2]+" "+line[3]+" "+line[-1]+'\n')
            cat2+=1

        if (i/len(lines)*100)%5==0:
            print('完成度：',i/len(lines)*100,'%')
        i+=1
    # print('treeNum: ',treeNum)
    print('tree提取物占比: ',(cat2/i)*100,'%')
    # print('roadNum: ',roadNum)
    print('roof提取物占比: ',(cat1/i)*100,'%')

    f_in.close()
    p.close()


if __name__ == '__main__':
    data_cat('./Vaihingen3D_Traininig')

