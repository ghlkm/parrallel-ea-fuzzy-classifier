import matplotlib.pyplot as plt
from  mpl_toolkits.mplot3d import  Axes3D
import numpy as np
# fs=['ms_nsgaii_100g_gs_72c2.o846839',
#     'ms_nsgaii_100g_gs_36c2.o846844',
#     'ms_nsgaii_100g_gs_18c.o846824',
#     'ms_100g_gs_9.o846922',
#     'ms_100g_gs_4.o846923',
#     'ms_100g_gs_2.o846907',
#     'ms_100g_br_72.o846920',
#     'ms_nsgaii_100g_br_36c2.o846843',
#     'ms_nsgaii_100g_br_18c.o846822',
#     'ms_100g_br_9.o846918',
#     'ms_100g_br_4.o846917',
#     'ms_100g_br_2.o846916']
fs=[
    # 'ring_100g_br_2.o846960',
    # 'ring_100g_br_4.o846961',
    # 'ring_100g_br_9.o846962',
    # 'ring_100g_br_18.o846971',
    # 'ring_100g_br_36.o846965',
    # 'ring_100g_gs_2.o846972',
    # 'ring_100g_gs_4.o846973',
    # 'ring_100g_gs_9.o846974',
    # 'ring_100g_gs_18.o846970',
    # 'ring_100g_gs_36.o846968',
    'ring_100g_br_72.o846964',
    'ring_100g_gs_72.o846967',
]
for i in fs:
    lnum=1
    sort_t=0
    evol_t=0
    rules=[]
    evol_cnt=0
    sort_cnt=0

    with open(i, 'r')  as f:
        for line in f:
            ls=line.split(' ')
            if len(ls)==3:
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
            # if lnum > 1:
            #     if lnum <= 202:
            #         if lnum % 2 == 0:
            #             sort_t += float(line)
            #         else:
            #             evol_t += float(line.split(' ')[2])
            #     elif lnum <= 402:
            #         rules.append(line.split(','))
            # lnum += 1
    rules=np.array(rules, dtype=float)
    avgsort=sort_t/sort_cnt
    avgevol=evol_t/evol_cnt
    print(avgevol, avgsort, '];')
    fig=plt.figure()
    ax=fig.add_subplot(111, projection='3d')
    ax.scatter(rules[:, 0], rules[:, 1], rules[:, 2])
    ax.view_init(azim=45)
    ax.set_xlabel('rule num')
    ax.set_ylabel('training error')
    ax.set_zlabel('testing error')
    # plt.show()
    name=i.split('.')
    ax.set_title(name[0])

    fig.savefig(name[0]+'.png')