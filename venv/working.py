import numpy as np
import random
import time
import copy
import math
import sys
from sklearn.decomposition import PCA
from multiprocessing import Process


def _gen_data_info(loc_instances, loc_label_set):
    data_info = dict()
    data_info["inst"] = loc_instances.shape[0]
    data_info["attr"] = loc_instances.shape[1]
    data_info["lnum"] = len(loc_label_set)  # label number
    return data_info


def _gen_fuzzy_set():
    fuzzy_set = dict()  # format: (set order, set number)
    fuzzy_set[0] = (0, 0)
    fuzzy_set[1] = (1, 2)
    fuzzy_set[2] = (2, 2)
    fuzzy_set[3] = (1, 3)
    fuzzy_set[4] = (2, 3)
    fuzzy_set[5] = (3, 3)
    fuzzy_set[6] = (1, 4)
    fuzzy_set[7] = (2, 4)
    fuzzy_set[8] = (3, 4)
    fuzzy_set[9] = (4, 4)
    fuzzy_set[10] = (1, 5)
    fuzzy_set[11] = (2, 5)
    fuzzy_set[12] = (3, 5)
    fuzzy_set[13] = (4, 5)
    fuzzy_set[14] = (5, 5)
    return fuzzy_set


def _train_mode(mode, _data, _label):
    if mode == '5CV':
        tmp = int(_data.shape[0] * 0.8)
    elif mode == '10CV':
        tmp = int(_data.shape[0] * 0.9)
    else:
        tmp = _data.shape[0] // 2
    return _data[:tmp], _data[tmp:], _label[:tmp], _label[tmp:]


# --------------------------下层函数与顶层函数分隔符-------------------------------------------
##########################################################################################


def to_membership(loc_data, fuzzy_sets, loc_data_info):
    """

    对于每个数据的每个维度，每个维度相对应的fuzzyset 的membership都是固定的，我们预先计算好
    here we have len(fuzzy_sets) = 15
    :param loc_data: 这个是训练数据， 类型应该是numpy.ndarray
    :param fuzzy_sets: 模糊集
    :param loc_data_info: 训练数据的信息
    :return: 举个例子 loc_mbs[i][j][k] 表示 $\mu_{fuzzy_set(k)}(x_ij)$,
             也就是第 i 个数据第 j 维对于 模糊集 k 的membership
    """
    loc_mbs = np.ndarray(shape=(loc_data_info["inst"], loc_data_info["attr"], len(fuzzy_sets)))
    for i in range(loc_data.shape[0]):
        for j in range(loc_data.shape[1]):
            xpi = loc_data[i][j]
            for fs_i in range(len(fuzzy_sets)):
                fs = fuzzy_sets[fs_i]
                if fs_i == 0:
                    # don't care condition
                    loc_mbs[i][j][fs_i] = 1
                else:
                    b = 1 / (fs[1] - 1)
                    a = (fs[0] - 1) * b
                    loc_mbs[i][j][fs_i] = max([1 - abs(a - xpi) / b, 0])
    return loc_mbs


def generate_probability(loc_data, fuzzy_sets, loc_data_info, loc_P_dc):
    """

    :param loc_data:
    :param fuzzy_sets:
    :param loc_data_info:
    :return: loc_mbs 就是membership
             loc_pmbs就是依据论文中的公式 14 生成的概率矩阵
             举个例子 loc_pmbs[i][j][k]表示 第 i 个数据 第 j 个维度 第k个模糊集的 累计概率
             如 如果 loc_pmbs[i][j] = [0.8, 0.9, 0.9, 0.95, 1, 1]
                那么don't care（模糊集 0 ） 的概率是0.8，
                模糊集 1 的概率是0.9 -0.8 = 0.1
                模糊集 2 的概率是0.9 -0.9 = 0
                模糊集 3 的概率是0.95 -0.9 = 0.05
                模糊集 4 的概率是1 -0.95 = 0.05
                模糊集 5 的概率是1 -1 = 0
    """
    # 我们在初始化种群时要初始化rule时， 根据概率生成
    # 这个概率是根据某些数据对于所有fuzzy set的一个membership加权得到 这个公式是
    loc_P_c = loc_P_dc
    loc_mbs = to_membership(loc_data, fuzzy_sets, loc_data_info)
    loc_pmbs = np.zeros(shape=loc_mbs.shape)
    for i in range(loc_pmbs.shape[0]):
        for j in range(loc_pmbs.shape[1]):
            # 忽略don't care 相加， don't care有固定的概率被选中, 这个概率是 P_dc (P don't care)
            tmp = sum(loc_mbs[i][j][1:])
            loc_pmbs[i, j, 0] = loc_P_dc  # probability of don't care
            for k in range(1, loc_pmbs.shape[2]):
                # P_c = 1- P_dc, 这个代表了care的概率， 也就是除了don't care这个模糊集其他模糊集概率的总和
                loc_pmbs[i, j, k] = loc_pmbs[i, j, k - 1] + loc_mbs[i, j, k] / tmp * loc_P_c
    return loc_mbs, loc_pmbs


# --------------------------预处理函数与周期性函数分隔符---------------------------------------
##########################################################################################


def compatibility_grade(rule, xp_i, loc_mbs):
    """
        # 这个值其实是固定的，但我们不预先计算是因为时间复杂度太大太大，这也是为什么我们采用遗传来演化
    # 其中 时间复杂度为 O( (nm)^len(fuzzy set) ), 其中 n是数据的维度， m是数据量
    rule 这个rule是纯粹的rule, 只是一个list, 这个list只包含rule的 antecedent part, 也就是模糊集
    raw rule,  antecedent part
    :param rule:
    :param xp_i:
    :param loc_mbs:
    :return:
    """
    result = 1
    for i in range(len(rule)):
        result *= loc_mbs[xp_i][i][rule[i]]
    return result


# 这是每个rule 本身的性质， 它对每一个数据都有一个win
# 对于每个rule-class 对，它都有一个confident
# 这些返回值会作为rule的一部分存储在rule_c , rule_c表示一个完整的rule
def to_cf_q(rule_c, loc_data_info, label, loc_mbs):
    """

    :param rule_c:
    :param loc_data_info:
    :param label:
    :param loc_mbs:
    :return:
    """
    """
    这里对应了论文的公式 10
    :param rule_c:= a complete rule [[antecedent parts],cgs, Cq, CFq, correct]
            cgs 是一个list, cgs[i]表示对第 i 个训练数据， 它的compatibility grade,
            Cq
            CFq
            correct, 这个规则正确分类了多少个数据， 这个correct 在测试测试数据时有点不一样，
                     具体不一样在测试数据的correct是一个list包含了这个rule识别正确的数据的下标
                     在训练时为了效率， 只存储正确识别了多少个数据
                     对于reject或者识别错误的数据error, 另外存储， 这个可以看win_rule这个函数
    """
    rule = rule_c[0]  # antecedent parts
    cgs = []
    for xp_i in range(loc_data_info['inst']):
        cgs.append(compatibility_grade(rule, xp_i, loc_mbs))
    h = np.zeros(loc_data_info["lnum"])
    for cg_i in range(len(cgs)):
        cg = cgs[cg_i]
        h[label[cg_i]] += cg
    c_q = max(list(range(loc_data_info["lnum"])), key=lambda i: h[i])
    tmp = sum(h)
    if tmp != 0:
        cf_q = (2 * h[c_q] - tmp) / tmp
    else:
        cf_q = 0
    return cgs, c_q, cf_q


def win_rule(rules, xp_i, reject, error, loc_label):
    """
    两个目标
    f1: maximum correct
    f2: minimum rule set len

    S, a fuzzy set, [ruleset, fitness, total correct, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]
    # 这里我们要解决几个问题，有如下情景
    1. 只有一个rule获胜
    2. 有多个rule获胜， 每个rule指向的类不一样
    3. 有多个rule获胜， 每个rule指向的类一样,

    论文里的处理办法是对于第 2 种直接认定该数据不可分类
    对于第三种论文并没有详细说明， 我们这里只对第一个rule的correct +1
    另外我们只对比了前两个weight
    """
    # u_{Aq}(x_p)  * CF_q
    win_weights = []
    for rule_i in range(len(rules)):
        rule = rules[rule_i]
        # 这里的 rule[1][xp_i] 就是公式 11 的结果
        win_weights.append((rule[1][xp_i], rule_i))
    win_weights = sorted(win_weights, key=lambda i: i[0], reverse=True)
    # only consider the first one and second one
    fir = win_weights[0]
    if fir[0] <= 0:
        # reject
        reject.append(xp_i)
    elif len(win_weights) > 1:
        sec = win_weights[1]
        if fir[0] == sec[0]:
            # see if their class is the same
            if rules[fir[1]][2] == rules[sec[1]][2]:
                # same class
                if rules[fir[1]][2] == loc_label[xp_i]:
                    if rules[fir[1]][4] > rules[sec[1]][4]:
                        rules[fir[1]][4] += 1
                    else:
                        rules[sec[1]][4] += 1
                else:
                    error.append(xp_i)
            else:
                # difference class, considered as rejected
                reject.append(xp_i)
        else:  # fir weight only the biggest
            if rules[fir[1]][2] == loc_label[xp_i]:
                rules[fir[1]][4] += 1
            else:
                error.append(xp_i)
    else:
        # a valid rule
        if rules[fir[1]][2] == loc_label[xp_i]:
            rules[fir[1]][4] += 1
        else:
            error.append(xp_i)


def evaluate(fuzzy_set_c, loc_label, loc_data_info, loc_mbs, fitness):
    """
    fuzzy_set_c S, a fuzzy set, [ruleset, fitness, totalcorrect, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]  # win = cgs * CFq
    对于新生成的每个rule set 我们都补充完整它的信息，比如correct, Cq, Cfq等
    """
    rule_set = fuzzy_set_c[0]
    reject_i = []
    error_i = []
    for rule in rule_set:
        cgs, c_q, cf_q = to_cf_q(rule, loc_data_info, loc_label, loc_mbs)
        rule[1] = [i * cf_q for i in cgs]  # win
        rule[2] = c_q
        rule[3] = cf_q
        rule[4] = 0
    for xp_i in range(loc_data_info['inst']):
        win_rule(rule_set, xp_i, reject_i, error_i, loc_label)
    # drop the meaningless fuzzy set
    retain_set = []
    for rule_i in range(len(rule_set)):
        if not (rule_set[rule_i][3] <= 0 or rule_set[rule_i][4] <= 0):  # CFq <= 0
            retain_set.append(rule_i)
        # else:
        #     if random.random() < 0.5:
        #         retain_set.append(rule_i)
    # print(len(delete_set), len(ruleset))
    rule_set = [rule_set[i] for i in retain_set]
    # print(len(ruleset))
    if len(rule_set) == 0:
        return [[], 0, 0, 0], None, None
    fuzzy_set_c[0] = rule_set
    fuzzy_set_c[2] = sum([rule[4] for rule in rule_set])
    fuzzy_set_c[3] = len(rule_set)
    fuzzy_set_c[1] = fitness(fuzzy_set_c)  # a rule = 10 patterns
    return fuzzy_set_c, reject_i, error_i


# -------------------------------------周期运行函数与用户调参函数分割线-------------------------------
#################################################################################################


def gen_fittness_fun(loc_args):
    weight_vector = loc_args['weight-vector']

    def f1(indiv):
        """
        single objective
        :param indiv: indiv[2]
        :return:
        """
        return weight_vector[0] * indiv[2] + weight_vector[1] * indiv[3]

    def f2(indiv):
        """
        multi-objective
        :param indiv:
        :return:
        """
        return 0

    if loc_args['multi']:
        return f2
    else:
        return f1


# ------------------------------------GA-------------------------------------------------------
##############################################################################################

def _singel_population_sort(pop, loc_N_pop):
    return sorted(pop, key=lambda i: i[1], reverse=True)


def p_gen_ruleset(mr_x_i, loc_Pmbs):
    """
    mr_x_i: 根据这些训练数据根据公式14生成新的rule
    :param mr_x_i:
    :param loc_Pmbs:
    :return:
    """
    # 这里是为了生成rule set
    rules = []
    for i in mr_x_i:
        # ante = np.zeros(shape=g_data_info['attr'], dtype=int)
        ante = []
        for j in range(loc_Pmbs.shape[1]):
            r = random.random()
            p = loc_Pmbs[i][j]
            for k in range(len(p)):
                if p[k] > r:
                    ante.append(k)
                    break
        # assert len(ante) == loc_data_info['attr']
        rules.append([ante, 0, 0, 0, 0])
    return [rules, 0, 0, 0, 0]


def pg_rule_set(loc_pop_size, loc_label, loc_data_info, loc_mbs, loc_n_rule, loc_Pmbs, fitness):
    """
    依概率（公式14生成种群）
    :param loc_pop_size:
    :param loc_label:
    :param loc_data_info:
    :param loc_mbs:
    :param loc_n_rule:
    :param loc_Pmbs:
    :return:
    """
    pop = []
    while len(pop) != loc_pop_size:
        x_i = random.sample(range(loc_data_info["inst"]), loc_n_rule)
        rule_set_c = p_gen_ruleset(x_i, loc_Pmbs)
        rule_set_c, _, _ = evaluate(rule_set_c, loc_label, loc_data_info, loc_mbs, fitness)
        if len(rule_set_c[0]) == 0:
            continue
        pop.append(rule_set_c)
    return pop


def binary_tournament(pop):
    # TODO
    a, b = random.sample(range(len(pop)), 2)
    return pop[a], pop[b]


def uniform_rule_crossover(ruleset, howmany):
    """
    Michigan的crossover
    :param ruleset:
    :param howmany:
    :return:
    """
    size = len(ruleset)
    rules = []
    for i in range(howmany):
        # rule = [[], 0, 0, 0, 0]
        # ante = np.zeros(shape=g_data_info['attr'], dtype=int)
        ante = []
        a, b = random.sample(range(size), 2)
        if ruleset[a][4] > ruleset[b][4]:
            parent1 = ruleset[a][0]
        else:
            parent1 = ruleset[b][0]
        a, b = random.sample(range(size), 2)
        if ruleset[a][4] > ruleset[b][4]:
            parent2 = ruleset[a][0]
        else:
            parent2 = ruleset[b][0]
        for k, j in zip(parent1, parent2):
            if random.random() < 0.5:
                ante.append(k)
            else:
                ante.append(j)
        rule = [ante, 0, 0, 0, 0]
        rules.append(rule)
    return rules


def _crossover(parent1, parent2):
    """
    主函数的crossover
    S, a fuzzy set, [ruleset, fitness, correct, ruleset_len]
    rule set := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]
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
    if len(n1_n2) > 40:
        random.shuffle(n1_n2)
        n1_n2 = n1_n2[:40]
    offspring = [[], 0, 0, 0, 0]
    for i in n1_n2:
        offspring[0].append(copy.deepcopy(parents[i[0]][i[1]]))
    return offspring


def _mutation(individual, loc_P_M):
    """
    主函数的 mutation
    S, a fuzzy set, [ruleset, fitness, correct, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq,correct]
    """
    for rule_c in individual[0]:
        rule = rule_c[0]
        for i in range(len(rule)):
            if random.random() < loc_P_M:
                rule[i] = random.randint(0, 14)
    return individual


def gen_population_sort(loc_args):
    if loc_args['selection'] == 'NSGA-II':
        return _parato
    else:
        return _singel_population_sort


def simf_gbml(rule_set_c, mr_x_i, loc_Pmbs, loc_label, loc_data_info, loc_mbs, fitness):
    """
    Michigan
    single Iteration of Michigan-style Fuzzy GBML algorithm


    S, a fuzzy set, [ruleset, fitness, totalcorrect, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]
    """
    # S, reject, error = evaluate(S)
    # mr_x_i = reject + error
    n_mr = len(mr_x_i)
    rule_set = rule_set_c[0]
    rule_set_len = len(rule_set)
    n_replace = math.ceil(rule_set_len * 0.2)
    # how many rules is used GA operation
    n_ga = int(max([n_replace // 2, n_replace - n_mr]))
    # new_rules = []
    random.shuffle(mr_x_i)
    mr_x_i = mr_x_i[:n_replace - n_ga]
    ga_new_set = p_gen_ruleset(mr_x_i, loc_Pmbs)[0]
    # new_rules.extend(ga_new_set)
    rule_set.extend(ga_new_set)
    # if rule_set_len > 1:
    rule_set.extend(uniform_rule_crossover(rule_set, n_ga))
    # else:
    #    new_rules.extend(copy.deepcopy(rule_set))
    rule_set_c[0] = rule_set
    rule_set_c, _, _ = evaluate(rule_set_c, loc_label, loc_data_info, loc_mbs, fitness)
    if len(rule_set_c[0]) == 0:
        return None
    return rule_set_c


def _parato(pop, N_pop):  # 快速帕累托排序
    def takecrow(elem):
        return elem[4]

    def cal_crow(correctlist, rulelist, layer):
        for i in layer:
            if correctlist.index(i[2]) == 0 or correctlist.index(i[2]) == len(correctlist) - 1 or rulelist.index(
                    i[3]) == 0 or rulelist.index(i[3]) == len(rulelist) - 1:
                i[4] = sys.maxsize
            else:
                i[4] = correctlist[correctlist.index(i[2]) - 1] - correctlist[correctlist.index(i[2]) + 1] + \
                       rulelist[rulelist.index(i[3]) - 1] - rulelist[rulelist.index(i[3]) + 1]
                i[4] = i[4] / len(layer)

    rpop = []
    rSet = []
    result = pop
    for i in pop:
        rpop.append([i, 0, []])
    for i in rpop:
        np.where(rpop[:][0][2])
        for j in rpop:
            if (j[0][2] > i[0][2] and j[0][3] <= i[0][3]) or (j[0][2] >= i[0][2] and j[0][3] < i[0][3]):
                i[1] += 1
            elif (j[0][2] < i[0][2] and j[0][3] >= i[0][3]) or (j[0][2] <= i[0][2] and j[0][3] > i[0][3]):
                i[2].append(j)
    F = []
    for i in rpop:
        if i[1] == 0:
            F.append(i)
    while len(F) != 0:
        temp2 = F
        F = []
        temp = []
        for i in temp2:
            temp.append(i[0])
            for j in i[2]:
                j[1] -= 1
                if j[1] == 0 and j not in F:
                    F.append(j)
        rSet.append(temp)
    for i in rSet:
        correctlist = []
        rulelist = []
        if len(i) + len(result) > N_pop:
            for j in i:
                correctlist.append(j[2])
                rulelist.append(j[3])
            rulelist.sort(reverse=True)
            correctlist.sort(reverse=True)
            cal_crow(correctlist, rulelist, i)
            i.sort(key=takecrow, reverse=True)
            for j in i:
                if len(result) < N_pop:
                    result.append(j)
                else:
                    break
            break
        else:
            result += i
    return result



def genetic_algorithm(loc_mbs, loc_Pmbs, loc_data_info, loc_label, loc_args, P_dc):
    """
        Hybird Multiobjective Fuzzy GBML Algorithm

    """
    """
           running parameter
    """
    # debug = True
    N_rule = 14
    N_pop = 200

    # P_dc = 0.8  # Probability of don't care
    # P_c = 1 - P_dc  # Probability of care
    # pop_size = 200  # population size
    P_mC = 0.9  # Probability of main part crossover
    # P_MiC = 0.9  # Probability of Michigan part crossover
    # stop_generation = 5000        # stop condition, after how many generation
    stop_generation = 1  # stop condition, after how many generation

    print(loc_args['file-name'], ' will stop at', stop_generation)
    print('P_dc:', P_dc)
    print(loc_args)
    P_mM = 1 / loc_data_info["attr"]  # Probability of main part mutation
    P_M = 1 / loc_data_info["attr"]  # P_MI_mutation
    # function pointer for GA
    mutation = _mutation
    crossover = _crossover
    fitness = gen_fittness_fun(loc_args)
    selection = gen_population_sort(loc_args)
    pop = pg_rule_set(20000, loc_label, loc_data_info, loc_mbs, N_rule, loc_Pmbs, fitness)  # probabilistic generation
    print("init finish")
    print(len(pop))
    for g in range(stop_generation):
        g_begin = time.time()
        offsprings = []
        while len(offsprings) != N_pop:
            # print(len(pop))
            parent1, parent2 = binary_tournament(pop)
            if random.random() < P_mC:
                offspring = crossover(parent1, parent2)
            else:
                offspring = copy.deepcopy(parent1)
            offspring = mutation(offspring, P_M)
            offspring, rj_patterns, er_patterns = evaluate(offspring, loc_label, loc_data_info, loc_mbs, fitness)
            if len(offspring[0]) == 0:
                continue
            if random.random() < 0.5:
                # consider they are the same
                mr_patterns = rj_patterns + er_patterns
                offspring = simf_gbml(offspring, mr_patterns, loc_Pmbs, loc_label, loc_data_info, loc_mbs, fitness)
                if offspring is None:
                    continue
            offsprings.append(offspring)
        pop.extend(offsprings)
        pop = selection(pop, N_pop)
        pop = pop[:N_pop]
        if g % 10 == 0:
            print(g, ": ", time.time() - g_begin)
    return pop  # return all non-dominated results


# --------------------------------以下代码是测试集所需--------------------
#########################################################################
# 这个值其实是固定的，但我们不预先计算是因为时间复杂度太大太大，这也是为什么我们采用遗传来演化
# 其中 时间复杂度为 O( (nm)^len(fuzzy set) ), 其中 n是数据的维度， m是数据量
def tst_cg(rule, xp_i, tst_mbs):
    """
    raw rule, rule[0] is antecedent part
    """
    result = 1
    for i in range(len(rule)):
        result *= tst_mbs[xp_i][i][rule[i]]
    return result


# 这是每个rule 本身的性质， 它对每一个数据都有一个win
# 比如第i个数据， 那么它的win就是cgs[i] * CFq
# 对于每个rule-class 对，它都有一个confident
# 这些返回值会作为rule的一部分存储在rule
def tst_to_cf_q(rule_c, tst_mbs, tst_dat):
    """
    rule_c:= a complete rule [[antecedent parts],cgs, Cq, CFq]
    """
    rule = rule_c[0]  # antecedent parts
    cgs = []
    for xp_i in range(tst_dat.shape[0]):
        cgs.append(tst_cg(rule, xp_i, tst_mbs))
    return cgs


def tst_win_rule(rules, xp_i, reject, error, tst_label):
    """
    f1: maximum correct
    f2: minimun ruleset_len

    S, a fuzzy set, [ruleset, fitness, correct, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]
    # 这里我们要解决几个问题，有如下情景
    1. 只有一个rule获胜
    2. 有多个rule获胜， 每个rule指向的类不一样
    3. 有多个rule获胜， 每个rule指向的类一样

    论文里的处理办法是对于第 2 种直接认定该数据不可分类
    """
    # u_{Aq}(x_p)  * CF_q
    win_weights = []
    for rule_i in range(len(rules)):
        rule = rules[rule_i]
        win_weights.append((rule[1][xp_i], rule_i))
    win_weights = sorted(win_weights, key=lambda i: i[0], reverse=True)
    # only consider the first one and second one
    fir = win_weights[0]
    if fir[0] <= 0:
        # reject
        reject.append(xp_i)
    elif len(win_weights) > 1:
        sec = win_weights[1]
        if fir[0] == sec[0]:
            # see if their class is the same
            if rules[fir[1]][2] == rules[sec[1]][2]:
                # same class
                if rules[fir[1]][2] == tst_label[xp_i]:
                    rules[fir[1]][4].append(xp_i)
                    # rules[sec[1]][4] += 1
                else:
                    error.append(xp_i)
            else:
                # difference class, considered as rejected
                reject.append(xp_i)
        else:  # fir weight only the biggest
            # print(xp_i)
            if rules[fir[1]][2] == tst_label[xp_i]:
                rules[fir[1]][4].append(xp_i)
            else:
                error.append(xp_i)
    else:
        # a valid rule
        if rules[fir[1]][2] == tst_label[xp_i]:
            rules[fir[1]][4].append(xp_i)
        else:
            error.append(xp_i)


def tst_evaluate(fuzzy_set_c, tst_mbs, tst_dat, tst_label):
    """
    fuzzy_set_c S, a fuzzy set, [ruleset, fitness, totalcorrect, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]  # win = cgs * CFq ,对于每一个数据,它的win
    """
    rule_set = fuzzy_set_c[0]
    reject_i = []
    error_i = []
    for rule in rule_set:
        cgs = tst_to_cf_q(rule, tst_mbs, tst_dat)
        rule[1] = [i * rule[3] for i in cgs]  # win
        rule[4] = []  # store those patterns that been correct
    for xp_i in range(tst_dat.shape[0]):
        tst_win_rule(rule_set, xp_i, reject_i, error_i, tst_label)
    fuzzy_set_c[0] = rule_set
    fuzzy_set_c[2] = sum([len(rule[4]) for rule in rule_set])
    fuzzy_set_c[3] = len(rule_set)
    fuzzy_set_c[1] = 0
    return fuzzy_set_c, reject_i, error_i


def main(loc_args):
    begin = time.time()
    P_dc = 0.8
    g_data = np.genfromtxt(loc_args['file-name'], delimiter=loc_args['delimiter'])[:, :-1]
    labels = np.genfromtxt(loc_args['file-name'], delimiter=loc_args['delimiter'], usecols=[-1], dtype=str)
    randomize = random.sample(range(g_data.shape[0]), g_data.shape[0])
    g_data = g_data[randomize]
    labels = labels[randomize]
    # map label
    g_fuzzy_set = _gen_fuzzy_set()
    label_id = 0
    label_set = set(labels)
    label_dict = dict()
    tmp = np.zeros(labels.shape[0], dtype=int)
    for i in label_set:
        label_dict[i] = label_id
        label_id += 1
    for i in range(g_data.shape[0]):
        tmp[i] = label_dict[labels[i]]
    print(label_dict)
    labels = tmp
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
    train_data, tst_data, label, tst_label = _train_mode(loc_args['mode'], g_data, labels)

    instances = train_data

    g_data_info = _gen_data_info(instances, label_set)

    """
    mbs:  memberships
    Pmbs: probability of memberships
    """
    mbs, Pmbs = generate_probability(instances, g_fuzzy_set, g_data_info, P_dc)
    rs = 0
    random.seed(rs)
    np.random.seed(rs)
    print('random seed:', rs)
    print('train data:', train_data.shape)
    print('test data:', tst_data.shape)
    print('label number:', len(label_set))
    pop = genetic_algorithm(mbs, Pmbs, g_data_info, label, loc_args, P_dc)

    instances = tst_data
    g_data_info = _gen_data_info(instances, label_set)
    tst_mbs = to_membership(tst_data, g_fuzzy_set, g_data_info)
    for best in pop[:200]:
        print('#', best[2] / train_data.shape[0], loc_args['file-name'], loc_args['selection'])  # total correct
        tst_evaluate(best, tst_mbs, tst_data, tst_label)
        this_best = best
        # fitness, correct, ruleset len
        print('$', this_best[2] / tst_data.shape[0], this_best[3], loc_args['file-name'], loc_args['selection'])
        for i in this_best[0]:
            print(i[0], i[2], i[3])
    print('finish at:', time.time() - begin)

class passdictobj:
    def __init__(self, d):
        self.dict = d



def main():
    return 0


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
    gas = GA.gas
    setting={
        'debug': True,
        'N_rule': 14,
        'N_pop' : 200,
        'P_dc' : 0.8,  # Probability of don't care
        'P_c' : 1 - P_dc,  # Probability of care
        'pop_size' : 200,  # population size
        'P_mC' : 0.9 , # Probability of main part crossover
        'P_MiC' : 0.9,  # Probability of Michigan part crossover
        'stop_generation' : 5000        # stop condition, after how many generation
    }
    args = {'file-name': None,
            'delimiter': ',',
            'PCA': False,
            'mode': '5CV',
            'multi': True,
            'normalize': True,
            'objective': None,  # function pointer
            'selection': 'NSGA-II',  # function pointer, singgle target
            'mutation': None,  # TODO for now, only one mutation method
            'weight-vector': (10, 0)}



