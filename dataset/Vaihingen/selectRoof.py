import sys
f_in=open('./Vaihingen3D_Traininig.pts')
p=open('./selectRoof.pts','w')
lines=f_in.readlines()
print(len(lines))
i=1
RoofNum=1

'''
0 Powerline
1 Low vegetation
2 Impervious surfaces
3 Car
4 Fence/Hedge
5 Roof
6 Facade
7 Shrub
8 Tree
'''

for line in lines:
    line=line.split(' ')
    
    
    # print(line[0])
    
    if line[-1]=='5\n':  # Roof
        print('cartory: ',line[-1],'\n')
        p.write(line[0]+' '+line[1]+' '+line[2]+" "+line[3]+'\n')
        RoofNum+=1
    # p.write("\n")
    print('完成度：',i/len(lines)*100,'%')
    i+=1
print('RoofNum: ',RoofNum)
print('提取物占比: ',RoofNum/i*100,'%')



f_in.close()
p.close()