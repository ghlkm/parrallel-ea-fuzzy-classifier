import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D
import numpy as np
import os
import ast
import sys
fs=os.listdir()


def nsga_ii(pop):  # 快速帕累托排序
    def dominate(one, other):
        if one[0] <= other[0] and one[1] <= other[1]:
            if one[0] < other[0] or one[1] < other[1]:
                return True
        return False
    # 这部分是多级排序所需的部分
    dominated_num = dict()
    dominate_list = dict()
    """ init """
    for index, value in enumerate(pop):
        dominated_num[index] = 0
        dominate_list[index] = []

    """ record dominated by how many, dominate who """
    for index, value in enumerate(pop):
        for index2, value2 in enumerate(pop[index + 1:]):
            if dominate(value, value2):
                dominated_num[index2 + index + 1] += 1
                dominate_list[index].append(index2 + index + 1)  # shadow copy
            elif dominate(value2, value):
                dominated_num[index] += 1
                dominate_list[index2 + index + 1].append(index)  # shadow copy

    # 这部分应该就是多级排序
    # list 只存储id 不用set是因为可以保证唯一性
    """ multi-level """
    F = [value for i, value in enumerate(pop) if dominated_num[i] == 0]
    return F



def getpf(pf):
    return nsga_ii(pf)


def __calhy__(rules):
    rules=np.array(rules)
    x=list(rules[:, 0])
    y=list(rules[:, 1])
    res=0.0
    x.insert(0, 0)
    y.insert(0, y[0])
    x.append(1)
    y.append(0)
    l=len(x)-1
    for i in range(l):
        assert x[i]<=1 and y[i]<=1
        res += (x[i+1] - x[i]) * (1 - y[i])
    return res


def calhy(rules):
    """
    1. find each time's pareto front
    2. calculate hy
    3. calculate avg

    :param rules:each element [rulenum, training error, testting error] or None

    :return:
    """
    parato_front=[]
    result=[]
    for i in rules:
        if i is None:
            """
            new time
            """
            if parato_front:
                """
                have begin
                
                deal with result
                
                calculate hv
                """
                parato_front=getpf(parato_front)
                result.append(__calhy__(parato_front))
            parato_front=[]
        else:
            parato_front.append(i)
    """
    the last time
    """
    parato_front = getpf(parato_front)
    parato_front = sorted(parato_front)
    result.append(__calhy__(parato_front))
    return sum(result)/len(result)



class rs:
    def __init__(self):
        self.avgmigr=0
        self.avggen=0
        self.dic=None
        self.dicstr=None
        self.rules=[]

    def __str__(self):
        result=''
        result+='| '+str(self.dic['migration_fre'])
        result+=' | '+str(self.dic['migration_num'])
        result += ' | ' + str(self.dicstr['hy'])
        result+=' | '+('%.4f' % self.dicstr['avgmigr'])
        if self.dic['copy_migration']:
            result+=' | '+'copy'
        else:
            result+=' | '+'cut'
        result += ' |'
        return result


    def __lt__(self, other):
        if self.dic['train-data'] < other.dic['train-data']:
            return True
        elif self.dic['train-data'] > other.dic['train-data']:
            return False
        elif self.dic['copy_migration'] == True and other.dic['copy_migration'] == False:
            return True
        elif self.dic['copy_migration'] == False and other.dic['copy_migration'] == True:
            return False
        elif self.dic['migration_fre'] < other.dic['migration_fre']:
            return True
        elif self.dic['migration_fre'] > other.dic['migration_fre']:
            return False
        elif self.dic['migration_num'] < other.dic['migration_num']:
            return True
        elif self.dic['migration_num'] > other.dic['migration_num']:
            return False
        else:
            return False



filedict={
    'Breast_w.csv':'Breast',
    'gesture.csv':'Gesture',
    'Glass.csv':'Glass',
    'sat.trn':'sat',
    'avila-tr.txt':'avila'
}

copydic={
    True:'copy',
    False:'cut'
}

results=[]

for filename in fs:
    """
    for each file
    """
    with open(filename, 'r') as f:
        gtime = 0
        gcnt = 0
        mtime=0
        mcnt=0
        rules=[]
        notemptyflag=False
        for line in f:
            """
            for each line
            """

            ls=line.split(' ')
            if len(ls) > 0:
                notemptyflag=True
                if ls[0] == 'read':
                    pass
                elif ls[0][0] == '{':
                    dic = ast.literal_eval(line)
                    if dic.get('migration'):
                        """
                        result info
                        """
                        info=filedict[dic['train-data']]+\
                             '_'+str(dic['migration_fre'])+'fre'+\
                             '_'+str(dic['migration_num'])+'num'+\
                             '_'+copydic[dic['copy_migration']]
                elif ls[0] == 'in':
                    """
                    single generation time
                    """
                    gtime+=float(ls[2])
                    gcnt+=1
                elif ls[0] == 'migration':
                    mtime+=float(ls[2])
                    mcnt+=1
                elif len(ls)==5:
                    rules.append([float(ls[0]), float(ls[2]), float(ls[4])])
                elif len(ls)>=2 and ls[1] == 'times':
                    rules.append(None)
        if notemptyflag:
            # rules = np.array(rules, dtype=float)
            if rules==[]:
                print(filename)
                continue
            # hy=calhy(rules)
            avgmigr = mtime/mcnt
            avggen = gtime/gcnt
            printdict={'avgmigr':avgmigr, 'avggen':avggen, 'hy':0, 'info':info}
            r=rs()
            r.dic=dic
            r.dicstr=printdict
            r.rules=rules
            results.append(r)

results=sorted(results)

pre_name = results[0].dic['train-data']
datagroup=[]
for i in results:
    if i.dic['train-data']!=pre_name:
        maxrulenum=max([max([j[0] for j in k.rules if j is not None]) for k in datagroup])
        print('maxrule num of ', pre_name, 'is', maxrulenum)
        for k in datagroup:
            for j, val in enumerate(k.rules):
                if val is not None:
                    k.rules[j][0]/=maxrulenum

        for k in datagroup:
            k.dicstr['hy']=calhy(k.rules)

        pre_name = i.dic['train-data']
        datagroup = [i]
    else:
        datagroup.append(i)
maxrulenum=max([max([j[0] for j in k.rules if j is not None]) for k in datagroup])
print('maxrule num of ', pre_name, 'is', maxrulenum)

for k in datagroup:
    for j, val in enumerate(k.rules):
        if val is not None:
            k.rules[j][0]/=maxrulenum

for k in datagroup:
    k.dicstr['hy']=calhy(k.rules)


"""
| dataset: 6 cores |
| fre | num | avg hy | avg migration time | copy/cut |
"""
pre_name = results[0].dic['train-data']
print(filedict[pre_name], '\r\n')
print('| fre | num | avg hy | avg migration time | copy/cut | ')
print('|-----|-----|--------|--------------------|----------|')
datagroup=[]
for i in results:
    if i.dic['train-data']!=pre_name:
        print('\r\n')
        pre_name = i.dic['train-data']
        print(filedict[pre_name], '\r\n')
        print('| fre | num | avg hy | avg migration time | copy/cut | ')
        print('|-----|-----|--------|--------------------|----------|')
        print(i)
    else:
        print(i)
