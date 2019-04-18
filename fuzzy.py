import numpy as np
import random
import time
import copy
import math
import sys


def _gen_fuzzy_set():
    fuzzy_set = np.array([None for _ in range(15)],dtype=tuple)  # format: (set order, set number)
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
    cgs = np.zeros(loc_data_info['inst'])
    for xp_i in range(loc_data_info['inst']):
        cgs[xp_i] = compatibility_grade(rule, xp_i, loc_mbs)
    h = np.zeros(loc_data_info["lnum"])
    for cg_i in range(len(cgs)):
        cg = cgs[cg_i]
        h[label[cg_i]] += cg
    c_q = np.argmax(h)
    tmp = np.sum(h)
    if tmp != 0:
        cf_q = (2 * h[c_q] - tmp) / tmp
    else:
        cf_q = 0
    # 这里cf_q可能小于 0
    return cgs, c_q, cf_q

def win_rule(rules, xp_i, reject, error, loc_label):
    """
    两个目标
    f1: maximum correct
    f2: minimum rule set len
    S, a fuzzy set, [rule_set, fitness, total correct, ruleset_len]
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
    win_rule_index = np.argmax(rules[:][1][xp_i])

    # only consider the first one and second one
    fir = rules[win_rule_index][1][xp_i]
    if fir <= 0:
        # reject
        reject.append(xp_i)
    elif len(rules[1]) > 1:
        tmp = rules[win_rule_index][1][xp_i]
        rules[win_rule_index][1][xp_i] = 0
        win_rule_index2 = np.argmax(rules[:][1][xp_i])
        rules[win_rule_index][1][xp_i] = tmp
        sec = rules[win_rule_index2][1][xp_i]
        if rules[fir][1][xp_i] == rules[sec][1][xp_i]:
            # see if their class is the same
            if rules[fir][2] == rules[sec][2]:
                # same class
                if rules[fir][2] == loc_label[xp_i]:
                    if rules[fir][4] > rules[sec][4]:
                        rules[fir[1]][4] += 1
                    else:
                        rules[sec[1]][4] += 1
                else:
                    error.append(xp_i)
            else:
                # difference class, considered as rejected
                reject.append(xp_i)
        else:  # fir weight only the biggest
            if rules[fir][2] == loc_label[xp_i]:
                rules[fir][4] += 1
            else:
                error.append(xp_i)
    else:
        # a valid rule
        if rules[fir][2] == loc_label[xp_i]:
            rules[fir][4] += 1
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
    rule_set=[r for r in rule_set if r[3]>0 and r[4]>0]
    if len(rule_set) == 0:
        return [[], 0, 0, 0], None, None
    fuzzy_set_c[0] = rule_set
    fuzzy_set_c[2] = sum([rule[4] for rule in rule_set])
    fuzzy_set_c[3] = len(rule_set)
    fuzzy_set_c[1] = fitness(fuzzy_set_c)  # a rule = 10 patterns
    return fuzzy_set_c, reject_i, error_i


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
        ante = np.zeros(loc_Pmbs.shape[1])
        for j in range(loc_Pmbs.shape[1]):
            r = random.random()
            p = loc_Pmbs[i][j]
            for k in range(len(p)):
                if p[k] > r:
                    ante[j]=k
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