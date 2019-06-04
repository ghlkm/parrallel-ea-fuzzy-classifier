import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D
import numpy as np
import os

fs=os.listdir()
# for i in fs:
#     with open(i, 'r')  as f:
#         name=''
#         for line in f:
#             ls=line.split(' ')
#             if ls[0]=='{\'migration\':':
#                 name+=ls[7].replace('\'', '').split('.')[0]
#                 name+='_'+ls[19].replace(',', '')+'g'
#                 name+='_'+ls[21].replace(',', '')+'Npop'
#                 if len(ls)==45:
#                     name+='_'+ls[40].replace(',', '')+'f'
#                     name+='_'+ls[42].replace(',', '')+'n'
#                     name+='_'+ls[44].replace(',', '').replace('}', '').replace('\n', '')
#                 else:
#                     name += '_' + ls[41].replace(',', '')+'f'
#                     name += '_' + ls[43].replace(',', '')+'n'
#                     name += '_' + ls[45].replace(',', '').replace('}', '').replace('\n', '')
#                 break
#     copyfile(i, name+'.out')



for i in fs:
    lnum=1
    sort_t=0
    evol_t=0
    rules=[]
    evol_cnt=0
    sort_cnt=0
    name=''
    with open(i, 'r')  as f:
        for line in f:
            ls=line.split(' ')
            if len(ls)==3 and ls[0]!='read':
                evol_t+=float(ls[-1])
                evol_cnt+=1
            elif len(ls)==5:
                rules.append((ls[0], ls[2], ls[4]))
                # rules.append(ls[2])
                # rules.append(ls[4])
            elif len(ls)==4:
                pass
            elif len(ls)==1:
                sort_t+=float(ls[0])
                sort_cnt+=1
            else:
                pass
                # if ls[0]=='{\'migration\':':
                #     name+=ls[7].replace('\'', '').split('.')[0]
                #     name+='_'+ls[19].replace(',', '')
                #     name+='_'+ls[21].replace(',', '')
                #     if len(ls)==45:
                #         name+='_'+ls[40].replace(',', '')+'f'
                #         name+='_'+ls[42].replace(',', '')+'n'
                #         name+='_'+ls[44].replace(',', '').replace('}', '').replace('\n', '')
                #     else:
                #         name += '_' + ls[41].replace(',', '')+'f'
                #         name += '_' + ls[43].replace(',', '')+'n'
                #         name += '_' + ls[45].replace(',', '').replace('}', '').replace('\n', '')
    rules=np.array(rules, dtype=float)
    avgsort=sort_t/sort_cnt
    avgevol=evol_t/evol_cnt
    print(avgevol, avgsort, '];')
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim((0, 17))
    ax.set_ylim((0, 0.6))
    # ax.set_zlim((0, 0.6))
    ax.scatter(rules[:, 0], rules[:, 1])
    # ax.view_init(azim=45)
    ax.set_xlabel('rule num')
    ax.set_ylabel('training error')
    # ax.set_zlabel('testing error')
    ax.set_title(i.split('.')[0])

    fig.savefig(i.split('.')[0]+'.png')