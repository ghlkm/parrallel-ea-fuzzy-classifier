import numpy as np
import random
import time
import copy
import math
import sys
from sklearn.decomposition import PCA
from multiprocessing import Process, Manager
import final

'''
  MOCCA
  设该数据有m个类，那么将population分为m个sub_population,记作specie
  species 是所有specie的集合
  第i个specie的元素是个结论项为i的rules的fuzzy_set

  评估方法:评估第i个specie内的元素，就选取其他species中最优的元素以及两个随机元素，组合得到3组不同的分类器，
  然后通过训练集和适应度函数算出，该元素组成的3组分类器的适应度/3即得该元素的适应度。

  species是共享的，每个进程都能读所有的species，但只能写自己进程对应的species
  '''
def fit(fuzzy_set,loc_mbs, loc_data_info, loc_label):
    '''
    适应度函数
    '''
    args = {
        'weight-vector':(10,0),
        'multi':0
    }
    final.evaluate(fuzzy_set,loc_label,loc_data_info,loc_mbs,final.gen_fittness_fun(args))

    return fuzzy_set[1]

def takeSecond(elem):
    return elem[1]

def best_tst(tst_specie,species):
    best_tst_set = []
    for i in species:
        if i != tst_specie:
            best_tst_set+=i[0][0]
    return best_tst_set

def random_tst(tst_specie,species):
    random_tst_set1 = []
    random_tst_set2 = []
    for i in species:
        if i != tst_specie:
            random_tst_set1+=i[random.randint(1,len(i)-1)][0]
    for i in species:
        if i != tst_specie:
            random_tst_set2+=i[random.randint(1,len(i)-1)][0]
    return random_tst_set1,random_tst_set2

def evaluate(tst_specie,species,loc_mbs, loc_data_info, loc_label):
    '''
    tst_specie:{{fuzzy_set1,fit},{fuzzy_set2,fit},......,{fuzzy_setN,fit}}要被评估的specie
    :param species: {specie1,specie2,.....,specieM},specie = {{fuzzy_set1,fit},{fuzzy_set2,fit},.....,{fuzzy_setN,fit}}
    fuzzy_set: {rule1,rule2,......,rulen}
    :return:
    '''
    best_tst_set = best_tst(tst_specie,species)
    random_tst_set1, random_tst_set2 = random_tst(tst_specie,species)
    for i in tst_specie:
        i[1] = 0
        fuzzy_set = best_tst_set + i[0]
        i[1] += fit([fuzzy_set,0,0,0],loc_mbs, loc_data_info, loc_label)
        fuzzy_set = random_tst_set1 + i[0]
        i[1] += fit([fuzzy_set,0,0,0],loc_mbs, loc_data_info, loc_label)
        fuzzy_set = random_tst_set2 + i[0]
        i[1] += fit([fuzzy_set,0,0,0],loc_mbs, loc_data_info, loc_label)
        i[1] /= 3
    tst_specie.sort(key=takeSecond,reverse=True)


def michigan_gbml(rule_set_c, mr_x_i, loc_Pmbs, loc_label, loc_data_info, loc_mbs, fitness):
    """
    Michigan
    single Iteration of Michigan-style Fuzzy GBML algorithm
    S, a fuzzy set, [ruleset, fitness, totalcorrect, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]
    """

    if len(rule_set_c[0]) == 0:
        return None
    return rule_set_c


def crossover(parent1, parent2):
    """
    rule set := [rules,fit]
    rule := [[antecedent parts], cgs * CFq, Cq, CFq, correct]
    """
    parents = (parent1[0], parent2[0])
    p1_len = len(parents[0])
    p2_len = len(parents[1])
    n1_i = random.sample(range(p1_len), random.randint(1, p1_len))
    n1 = [(0, i) for i in n1_i]
    n2_i = random.sample(range(p2_len), random.randint(1, p2_len))
    n2 = [(1, i) for i in n2_i]
    n1_n2 = n1 + n2
    # print(len(n1_n2))
    if len(n1_n2) > 10:
        random.shuffle(n1_n2)
        n1_n2 = n1_n2[:10]
    offspring = [[],0]
    for i in n1_n2:
        offspring[0].append(copy.deepcopy(parents[i[0]][i[1]]))
    return offspring


def mutation(child,loc_Pmbs,loc_data_info,loc_label,loc_mbs):
    x_i = random.sample(range(loc_data_info["inst"]),1)
    while True:
         rule = generate_rule(x_i[0], loc_Pmbs)
         cgs, c_q, cf_q = final.to_cf_q(rule, loc_data_info, loc_label, loc_mbs)
         rule[1] = [i * cf_q for i in cgs]  # win
         rule[2] = c_q
         rule[3] = cf_q
         rule[4] = 0
         if rule[3] <= 0:  # CFq <= 0
             continue
         if child[0][random.randint(0,len(child[0])-1)][2] != rule[2]:
             continue
         else:
             child[0][random.randint(0, len(child[0])-1)] = rule
             break


def gbml(id,species,gen,loc_mbs, loc_Pmbs, loc_data_info, loc_label):

    '''

    :return:
    '''
    tst_specie = species[id]
    popN = len(species[id])
    for i in range(gen):
        print(id ,'is in',i,' gen')
        offspring = []
        while len(offspring) != popN:
            parent1, parent2 = final.binary_tournament(tst_specie)
            child = crossover(parent1, parent2)
            if random.random() < 0.1: #密西根GA产生新的规则，但是该规则的结束项需和亲代相同
                mutation(child,loc_Pmbs,loc_data_info,loc_label,loc_mbs)
            offspring.append(child)
        tst_specie+=offspring
        evaluate(tst_specie,species,loc_mbs, loc_data_info, loc_label) ##评估排序
        tst_specie[:popN]
        species[id] = tst_specie

def generate_rule(mr_x_i, loc_Pmbs):
        # ante = np.zeros(shape=g_data_info['attr'], dtype=int)
    ante = []
    for j in range(loc_Pmbs.shape[1]):
        r = random.random()
        p = loc_Pmbs[mr_x_i][j]
        for k in range(len(p)):
            if p[k] > r:
                ante.append(k)
                break
    return  [ante, 0, 0, 0, 0]

def init_species_list(N_pop,N_rule,loc_mbs, loc_Pmbs, loc_data_info, loc_label):
    species_list = []
    species_dict = {}
    flag_dict = {}
    total = N_pop*N_rule
    for i in loc_label:
        species_dict[i] = 0
        flag_dict[i] = True
    x_i = random.sample(range(loc_data_info["inst"]),1)
    flag = True
    while flag:
        rule = generate_rule(x_i[0], loc_Pmbs)
        cgs, c_q, cf_q = final.to_cf_q(rule, loc_data_info, loc_label, loc_mbs)
        rule[1] = [i * cf_q for i in cgs]  # win
        rule[2] = c_q
        rule[3] = cf_q
        rule[4] = 0
        if rule[3] <= 0:  # CFq <= 0
            continue
        if species_dict[c_q] == 0:
            species_dict[c_q] = [rule,]
        elif flag_dict[c_q]:
            species_dict[c_q].append(rule)
            if len(species_dict[c_q]) == total:
                flag_dict[c_q] = False

        flag = False
        for i in flag_dict.items():
            flag = flag | i[1]
    for i in species_dict.keys():
        temp_list = []
        m = 0
        j = N_rule
        while j<=total:
            temp_list.append([species_dict[i][m:j],0])
            j+=N_rule
            m+=N_rule
        species_list.append(temp_list[:])
    return species_list


def CA(gen,pop,size,loc_mbs, loc_Pmbs, loc_data_info, loc_label):
    species_list = init_species_list(pop,size,loc_mbs,loc_Pmbs,loc_data_info,loc_label)
    for i in species_list:   #对species先估值排序
        evaluate(i,species_list,loc_mbs,loc_data_info,loc_label)
    with Manager() as manager:
        species = manager.list(species_list)
        p_list = []
        for i in range(len(species_list)):
            p = Process(target=gbml,args=(i,species,gen,loc_mbs, loc_Pmbs, loc_data_info, loc_label))
            p.start()
            print(p.pid,'is start')
            p_list.append(p)
        for res in p_list:
            res.join()
        result = []
        for i in species:
            result += i[0][0]
        result = [result,0,0,0]
        args = {
            'weight-vector': (10,0),
            'multi':0
        }
        final.evaluate(result,loc_label,loc_data_info,loc_mbs,final.gen_fittness_fun(args))
        print(result[2])
        return(result)

def main(loc_args):
    begin = time.time()
    P_dc = 0.8
    g_data = np.genfromtxt(loc_args['file-name'], delimiter=loc_args['delimiter'])[:, :-1]
    labels = np.genfromtxt(loc_args['file-name'], delimiter=loc_args['delimiter'], usecols=[-1], dtype=str)
    randomize = random.sample(range(g_data.shape[0]), g_data.shape[0])
    g_data = g_data[randomize]
    labels = labels[randomize]
    # map label
    g_fuzzy_set = final._gen_fuzzy_set()
    label_id = 0
    label_set = set(labels)
    label_dict = dict()
    tmp = np.zeros(labels.shape[0], dtype=int)
    for i in label_set:
        label_dict[i] = label_id
        label_id += 1
    for i in range(g_data.shape[0]):
        tmp[i] = label_dict[labels[i]]
    labels = tmp
    # drop na
    naise = ~np.isnan(g_data).any(axis=1)
    labels = labels[naise]
    g_data = g_data[naise]

    # pca or not
    if loc_args['PCA']:
        pca_nc = 0.95
        pca = PCA(n_components=pca_nc)
        print('PCA components', pca_nc)
        print('before PCA:', g_data.shape)
        g_data = pca.fit_transform(g_data)
        print('after PCA:', g_data.shape)
        print('PCA variance explanation:', pca.explained_variance_)
        # g_data = pca.transform(g_data)

    # normalize or not
    if loc_args['normalize']:
        print('normalize')
        # g_data = g_data / np.linalg.norm(g_data, axis=0)
        ming = np.amin(g_data, axis=0)
        maxg = np.amax(g_data, axis=0)
        rangeg = maxg - ming
        for i in range(len(g_data)):
            g_data[i] = (g_data[i] - ming) / rangeg

    # based on the mode, design which part is training, which is testing
    train_data, tst_data, label, tst_label = final._train_mode(loc_args['mode'], g_data, labels)

    instances = train_data

    g_data_info = final._gen_data_info(instances, label_set)
    """
    mbs:  memberships
    Pmbs: probability of memberships
    """
    mbs, Pmbs = final.generate_probability(instances, g_fuzzy_set, g_data_info, P_dc)
    rs = 0
    random.seed(rs)
    np.random.seed(rs)
    best = CA(5,4,1,mbs,Pmbs, g_data_info, label)
    print('best:',best)

if __name__ == '__main__':
    # read all data, including training data and testing data
    # global data
    files = [
        'Breast_W.csv',
        'sonar2.csv',
        'a0_0_pima-10tra.dat',
        'gesture.csv',
        'Glass.csv',
        'Sonar.dat',
        'wine_change.csv',
        'sonar3.csv'
    ]
    """
       running parameter
    """
    # debug = True
    # N_rule = 14
    # N_pop = 200
    # P_dc = 0.8  # Probability of don't care
    # P_c = 1 - P_dc  # Probability of care
    # pop_size = 200  # population size
    # P_mC = 0.9  # Probability of main part crossover
    # P_MiC = 0.9  # Probability of Michigan part crossover
    # stop_generation = 5000        # stop condition, after how many generation
    # stop_generation = 1  # stop condition, after how many generation
    args = {'file-name': 'sonar3.csv',
            'delimiter': ',',
            'PCA': False,
            'mode': '5CV',
            'multi': True,
            'normalize': True,
            'objective': None,  # function pointer
            'selection': 'NSGA-II',  # function pointer, singgle target
            'mutation': None,  # TODO for now, only one mutation method
            'weight-vector': (10, 0)}
    main(args)

