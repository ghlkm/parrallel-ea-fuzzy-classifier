import numpy as np
import random
import time
import copy
import math
import sys
# from sklearn.decomposition import PCA
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Manager
"""
@author ghlkm
"""

MASTER_SLAVE='master-slave'
RING='ring'
DYNAMIC='dynamic'

NSGA_II='nsga-ii'

NODIVISION='NO'

class Rule:
    def __init__(self, rule=None):
        if rule is None:
            self.rule = []
        else:
            self.rule = rule
        self.cgs_mul_cfq=None
        self.cq=0
        self.cfq=0
        self.correct=0
    def __len__(self):
        return len(self.rule)
    def __getitem__(self, item):
        return self.rule[item]
    def __setitem__(self, key, value):
        self.rule[key]=value
    def __lt__(self, other):
        return self.correct<other.correct
    def __gt__(self, other):
        return self.correct>other.correct
    def __eq__(self, other):
        return self.correct==other.correct
    def __ge__(self, other):
        return self.correct>=other.correct
    def __le__(self, other):
        return self.correct<=other.correct

class RuleSet:
    def __init__(self, rules=None):
        if rules:
            self.rules=rules
        else:
            self.rules=[]
        self.fitness=0
        self.objective=np.array([0, 0], dtype=[('rule_len', '<i4'), ('correct', '<i4')])
        self.dominated_num=0
        self.dominate_list=[]
    def __lt__(self, other):
        return self.fitness>other.fitness
    def __gt__(self, other):
        return self.fitness<other.fitness
    def __eq__(self, other):
        return self.fitness==other.fitness
    def __ge__(self, other):
        return self.fitness<=other.fitness
    def __le__(self, other):
        return self.fitness>=other.fitness
    def __len__(self):
        return len(self.rules)
    def __getitem__(self, item):
        return self.rules[item]
    def __iter__(self):
        return self.rules.__iter__()
    def append(self, item):
        self.rules.append(item)
    def extend(self, items):
        self.rules.extend(items)

def isSame(r1:RuleSet, r2:RuleSet):
    return r1.objective['rule_len'][0]==r2.objective['rule_len'][0] and \
           r1.objective['correct'][0] == r2.objective['correct'][0]
def inPop(r1:RuleSet, pop):
    for r2 in pop:
        if isSame(r1, r2):
            return True
    return False

def _gen_next_process_data(args, data, label):
    return data, label, data, label
def getWorker(args):
    if args['migration']==MASTER_SLAVE:
        return master_slave_master
    elif args['migration']==RING:
        return ring_ga
    else:
        return None

def _preprecess_data(loc_args, label_dict=None):
    """ read data """
    tr_data = np.genfromtxt(loc_args['train-data'], delimiter=loc_args['delimiter'])[:, :-1]
    tr_labels = np.genfromtxt(loc_args['train-data'], delimiter=loc_args['delimiter'], usecols=[-1], dtype=str)
    """ discard None data """
    naise = ~np.isnan(tr_data).any(axis=1)
    tr_labels = tr_labels[naise]
    tr_data = tr_data[naise]
    tr_size=tr_data.shape[0]
    if loc_args['test-data']:
        ts_data = np.genfromtxt(loc_args['test-data'], delimiter=loc_args['delimiter'])[:, :-1]
        ts_labels = np.genfromtxt(loc_args['test-data'], delimiter=loc_args['delimiter'], usecols=[-1], dtype=str)
        naise = ~np.isnan(ts_data).any(axis=1)
        ts_labels = ts_labels[naise]
        ts_data = ts_data[naise]
        data=np.vstack((tr_data, ts_data))
        labels=np.hstack((tr_labels, ts_labels))
    else:
        data=tr_data
        labels=tr_labels
    """ normalize """
    ming = np.amin(data, axis=0)
    maxg = np.amax(data, axis=0)
    rangeg = maxg - ming
    for i in range(len(data)):
        data[i] = (data[i] - ming) / rangeg

    tr_data=data[:tr_size]
    ts_data=data[tr_size:]
    tr_labels=labels[:tr_size]
    ts_labels=labels[tr_size:]
    if loc_args['test-data'] is None:
        tr_data, ts_data, tr_labels, ts_labels = _train_mode(run_which['mode'], tr_data, tr_labels)


    """ radomize data"""
    randomize = random.sample(range(tr_data.shape[0]), tr_data.shape[0])
    tr_data = tr_data[randomize]
    tr_labels = tr_labels[randomize]

    """ map label, convenient for computation """
    label_dict, tr_labels=_label_map(tr_labels, label_dict)
    label_dict, ts_labels=_label_map(ts_labels, label_dict)
    ts_labels=np.ndarray.astype(ts_labels, dtype=int)
    tr_labels=np.ndarray.astype(tr_labels, dtype=int)
    return tr_data, ts_data, tr_labels, ts_labels, label_dict

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

def to_membership(loc_data, fuzzy_sets):
    """

    对于每个数据的每个维度，每个维度相对应的fuzzyset 的membership都是固定的，我们预先计算好
    here we have len(fuzzy_sets) = 15
    :param loc_data: 这个是训练数据， 类型应该是numpy.ndarray
    :param fuzzy_sets: 模糊集
    :param loc_data_info: 训练数据的信息
    :return: 举个例子 loc_mbs[i][j][k] 表示 $\mu_{fuzzy_set(k)}(x_ij)$,
             也就是第 i 个数据第 j 维对于 模糊集 k 的membership
    """
    loc_mbs = np.ndarray(shape=(loc_data.shape[0], loc_data.shape[1], len(fuzzy_sets)))
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
def generate_probability(loc_data, fuzzy_sets, loc_P_dc):
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
    loc_P_c = 1-loc_P_dc
    loc_mbs = to_membership(loc_data, fuzzy_sets)
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

def gen_selection_fun(loc_args):
    w=loc_args['weight-vector']
    N_pop=loc_args['N_pop']
    def weight_vector(pop):
        return sorted(pop, key=lambda indiv:np.dot(w, indiv.objetive))
    def moea_d(pop):
        # todo
        return None
    def nsga_ii(pop):  # 快速帕累托排序
        def cak_crow(layer):
            """

            :param layer: (RuleSet, index)
            :return:
            """
            d = dict()
            for i in layer:
                d[i[1]] = 0
            layer = sorted(layer, key=lambda item: sum(item[0].objective['correct']))
            d[layer[0][1]] = sys.maxsize
            d[layer[-1][1]] = sys.maxsize
            for index, ruleset in enumerate(layer[1:-1]):
                d[ruleset[1]] += sum(layer[index + 1][0].objective['correct'] \
                                 - layer[index - 1][0].objective['correct'])
            layer = sorted(layer, key=lambda item: sum(item[0].objective['rule_len']))
            d[layer[0][1]] = sys.maxsize
            d[layer[-1][1]] = sys.maxsize
            for index, ruleset in enumerate(layer[1:-1]):
                d[ruleset[1]] += sum(layer[index + 1][0].objective['rule_len'] \
                                 - layer[index - 1][0].objective['rule_len'])
            return sorted(layer, key=lambda item: d[item[1]], reverse=True)

        def dominate(one, other):
            one = one.objective
            other = other.objective
            if (one['correct'] >= other['correct']).all() and (one['rule_len'] <= other['rule_len']).all():
                if (one['correct'] > other['correct']).all() or (one['rule_len'] < other['rule_len']).all():
                    return True
                else:
                    return False
            else:
                return False

        b = time.time()
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
        F = [(value, i) for i, value in enumerate(pop) if dominated_num[i] == 0]
        cnt = len(F)
        rSet = [F]
        layer_num=1
        while cnt < len(pop):
            Fi = []
            for value in F:
                for i in dominate_list[value[1]]:
                    dominated_num[i] -= 1
                    if dominated_num[i] <= 0:
                        pop[i].fitness=-layer_num
                        Fi.append((pop[i], i))
            layer_num+=1
            cnt += len(Fi)
            F = Fi
            rSet.append(Fi)
        result = []
        for layer in rSet:
            if len(layer) + len(result) <= N_pop:
                result += layer
            else:
                layer = cak_crow(layer)
                result += layer[:N_pop - len(result)]
        result = [i[0] for i in result]
        print(time.time()-b)
        return result
    if loc_args['objective']=='weight-vector':
        return weight_vector
    elif loc_args['objective']=='moea-d':
        return moea_d
    elif loc_args['objective']=='nsga-ii':
        return nsga_ii
    else:
        return None
def gen_mutation_fun(loc_args):
    loc_P_M=loc_args['P_M']
    def _mutation(individual:RuleSet):
        for rule in individual.rules:
            for i in range(len(rule)):
                if random.random() < loc_P_M:
                    rule[i] = random.randint(0, 14)
        return individual
    return _mutation

def random_two(pop):
    a, b = random.sample(range(len(pop)), 2)
    return (pop[a], pop[b])

def binary_tournament(pop):
    size = len(pop)
    a, b = np.random.randint(size), np.random.randint(size)
    if pop[a] > pop[b]:
        parent1 = pop[a]
    else:
        parent1 = pop[b]
    a, b = np.random.randint(size), np.random.randint(size)
    if pop[a] > pop[b]:
        parent2 = pop[a]
    else:
        parent2 = pop[b]
    return (parent1, parent2)

def parent_select(loc_args):
    return binary_tournament

def gen_crossover_fun(loc_args):
    """
    given pop, return offspring
    :param loc_args:
    :return:
    """
    select_parent_method=parent_select(loc_args)
    def _crossover(pop):
        parents = select_parent_method(pop)
        p1_len = len(parents[0])
        p2_len = len(parents[1])
        n1_i = random.sample(range(p1_len), random.randint(1, p1_len))
        n1 = [(0, i) for i in n1_i]
        n2_i = random.sample(range(p2_len), random.randint(1, p2_len))
        n2 = [(1, i) for i in n2_i]
        n1_n2 = n1 + n2
        if len(n1_n2) > 40:
            random.shuffle(n1_n2)
            n1_n2 = n1_n2[:40]
        offspring = RuleSet()
        for i in n1_n2:
            offspring.append(copy.deepcopy(parents[i[0]][i[1]]))
        return offspring
    return _crossover

def compatibility_grade(rule:Rule, xp_i, loc_mbs:np.ndarray):
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
    for i,j in enumerate(rule):
        result *= loc_mbs[xp_i, i, j]
    return result

def win_rule(rules:RuleSet, xp_i, reject, error, loc_label):
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
    for rule_i, rule in enumerate(rules):
        # 这里的 rule[1][xp_i] 就是公式 11 的结果
        win_weights.append((rule.cgs_mul_cfq[xp_i], rule_i))
    win_weights = sorted(win_weights, reverse=True)
    # only consider the first one and second one
    fir = win_weights[0]
    if fir[0] <= 0:
        # reject
        reject.append(xp_i)
    elif len(win_weights) > 1:
        sec = win_weights[1]
        if fir[0] == sec[0]:
            # see if their class is the same
            if rules[fir[1]].cq == rules[sec[1]].cq:
                # same class
                if rules[fir[1]].cq == loc_label[xp_i]:
                    if rules[fir[1]].correct > rules[sec[1]].correct:
                        rules[fir[1]].correct += 1
                    else:
                        rules[sec[1]].correct += 1
                else:
                    error.append(xp_i)
            else:
                # difference class, considered as rejected
                reject.append(xp_i)
        else:  # fir weight only the biggest
            if rules[fir[1]].cq == loc_label[xp_i]:
                rules[fir[1]].correct += 1
            else:
                error.append(xp_i)
    else:
        # a valid rule
        if rules[fir[1]].cq == loc_label[xp_i]:
            rules[fir[1]].correct += 1
        else:
            error.append(xp_i)
def _random_uniform_rule_(parent_rules):
    rs=np.random.randint(0, 2, size=len(parent_rules[0]))
    return np.array([parent_rules[j][i] for i, j in zip(range(len(parent_rules[0])), rs)])
def simf_gbml(rule_set:RuleSet, mr_x_i, loc_Pmbs:np.ndarray, loc_label, loc_data_info:dict, loc_mbs:np.ndarray):
    def uniform_rule_crossover(ruleset:RuleSet, howmany)->list:
        """
        Michigan的crossover
        :param ruleset:
        :param howmany:
        :return:
        """
        rules = []
        for i in range(howmany):
            parents=binary_tournament(ruleset)
            ante=_random_uniform_rule_(parents)
            rules.append(Rule(ante))
        return rules
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
    rule_set_len = len(rule_set)
    n_replace = math.ceil(rule_set_len * 0.2)
    # how many rules is used GA operation
    n_ga = int(max([n_replace // 2, n_replace - n_mr]))
    # new_rules = []
    random.shuffle(mr_x_i)
    mr_x_i = mr_x_i[:n_replace - n_ga]
    ga_new_set = p_gen_ruleset(mr_x_i, loc_Pmbs)
    rule_set.extend(ga_new_set.rules)
    rule_set.extend(uniform_rule_crossover(rule_set, n_ga))
    rule_set_c, _, _ = evaluate(rule_set, loc_label, loc_mbs, loc_data_info)
    return rule_set_c
def _label_map(labels, label_dict=None):
    if not label_dict:
        label_set = set(labels)
        label_dict = {i:j for i, j in zip(label_set, range(len(label_set)))}
    for i in range(len(labels)):
        labels[i]=label_dict[labels[i]]
    return label_dict, labels
def weight_vector(args, data, labels):
    """
    :param args:
        dataset
        verify-method
    :return:
    """
    init_population = args['init']
    crossover, mutation, evaluate = _gen_ga_function(args)
    pop=init_population(args['init_pop'], data, labels, evaluate)
    for g in range(args['stop_generation']):
        offsprings=[]

        while len(offsprings)!=args['N_pop']:
            offspring=crossover(pop)
            offspring=mutation(offspring)
            offspring, rj_patterns, er_patterns = evaluate(offspring, labels, mbs)
            if offspring is None:
                continue
            if random.random() < 0.5:
                # consider they are the same
                mr_patterns = rj_patterns + er_patterns
                offspring = simf_gbml(offspring, mr_patterns, Pmbs, labels, mbs, fitness)
                if offspring is None:
                    continue
            offsprings.append(offspring)
        pop.extend(offsprings)
        pop=sorted(pop)[:args['N_pop']]
    return pop


def ring_ga(args, data, labels):
    """ init parameter """
    args['lnum']=len(set(labels))
    print(args)

    args['lnum'] = len(set(labels))
    P_dc = args['P_dc']
    g_fuzzy_set = _gen_fuzzy_set()
    mbs, Pmbs = generate_probability(data, g_fuzzy_set, P_dc)

    """ structure """
    queue_ring_in=[Queue() for _ in range(args['core_num'])]
    queue_ring_out=[Queue() for _ in range(args['core_num'])]
    process_pool=[]
    for i in range(args['core_num']):
        p=Process(target=ring_created,
                    args=(args, data, labels, queue_ring_in[i], queue_ring_out[i], mbs, Pmbs,))
        process_pool.append(p)
    for p in process_pool:
        p.start()
    # itself also take part in
    results=[]
    can_put=np.array([True for _ in range(len(queue_ring_in))])
    while True:
        running=any(p.is_alive() for p in process_pool)
        if not running:
            break
        for i, q in enumerate(queue_ring_out):
            if not q.empty():
                get=q.get()
                if type(get)!=str:
                    if can_put[i-1]:
                        queue_ring_in[i-1].put(get)
                else:
                    while True:
                        get=q.get()
                        if type(get)==str:
                            break
                        results.append(get)
                    while not queue_ring_in[i].empty():
                        queue_ring_in[i].get()
                    can_put[i]=False
                    break

    for q,qq in zip(queue_ring_in, queue_ring_out):
        while not q.empty():
            q.get()
        while not qq.empty():
            qq.get()
        q.close()
        qq.close()
    for p in process_pool:
        p.join()
    for p in process_pool:
        p.terminate()
    selection=gen_selection_fun(args)
    tmp=[]
    for r in results:
        if not inPop(r, tmp):
            tmp.append(r)
    results=selection(tmp)[:min(args['N_pop'], len(tmp))]
    return results



def ring_migraion(pop, args, g, queue_in:Queue, queue_out:Queue):
    if  g%args['migration_fre'] == 0:
        tmpb = time.time()
        a=np.random.randint(0, len(pop)-args['migration_num'], size=args['migration_num'])
        for aa in a:
            if not args['copy_migration']:
                indiv=pop.pop(aa)
            else:
                indiv=pop[aa]
            if not queue_out.full():
                queue_out.put(indiv)
        for i in range(len(a)):
            if not queue_in.empty():
                get=queue_in.get()
                if not inPop(get, pop):
                    pop.append(get)
        print('migration ', time.time()-tmpb)


def ring_created(args, data, labels, queue_in:Queue, queue_out:Queue, mbs, Pmbs):
    """
    :param args:
        dataset
        verify-method
    :return:
    """
    """ init parameter"""
    N_pop=args['N_pop']

    """ init functions"""
    init_population = pg_rule_set
    crossover=gen_crossover_fun(args)
    mutation=gen_mutation_fun(args)
    selection = gen_selection_fun(args)
    P_m = args['P_m']
    """ init population """
    pop=init_population(args['init_N_pop'], labels, mbs, args['N_rule'], Pmbs, args)
    pop=selection(pop)
    """ begin evolution"""
    for g in range(args['stop_generation']):
        offsprings=[]
        b=time.time()
        while len(offsprings)!=N_pop:
            offspring=crossover(pop)
            offspring=mutation(offspring)
            offspring, rj_patterns, er_patterns = evaluate(offspring, labels, mbs, args)
            if offspring is None:
                continue
            if random.random() < P_m:
                # consider they are the same
                mr_patterns = rj_patterns + er_patterns
                offspring = simf_gbml(offspring, mr_patterns, Pmbs, labels, args, mbs)
                if offspring is None:
                    continue
            if inPop(offspring, pop):
                continue
            offsprings.append(offspring)
        pop.extend(offsprings)
        pop=selection(pop)[:args['N_pop']+args['migration_num']]
        ring_migraion(pop, args, g, queue_in, queue_out)
        pop=pop[:args['N_pop']]
        print('in main', time.time()-b)
        g+=1
    queue_out.put('begin')
    for i in pop:
        while queue_out.full():
            time.sleep(1e-3)
        queue_out.put(i)
    queue_out.put('end')


def master_slave_master(args, data, labels):
    """
    :param args:
        dataset
        verify-method
    :return:
    """

    """ init parameter """
    b=time.time()
    N_pop=args['N_pop']
    queue_in=Queue()
    queue_out=Queue()
    args['lnum']=len(set(labels))
    P_dc=args['P_dc']
    print(args)

    """ init functions"""
    init_population = pg_rule_set
    selection=gen_selection_fun(args)
    g_fuzzy_set = _gen_fuzzy_set()
    mbs, Pmbs = generate_probability(data, g_fuzzy_set, P_dc)
    print('gen mbs finish', time.time()-b)
    """ init population """
    pop=init_population(args['init_N_pop'], labels, mbs, args['N_rule'], Pmbs, args)
    print('init pop finish', time.time()-b)
    # 初始化了种群后计算适应度
    selection(pop)
    print('selection finish')
    """ """
    process_pool=[]
    for _ in range(args['core_num'] - 1):
        p = Process(target=master_slave_slave,
                    args=(queue_in, queue_out, args, labels, mbs, Pmbs,))
        process_pool.append(p)
        p.start()
    """ begin evolution"""
    print('begin to evolve')
    for g in range(args['stop_generation']):
        b=time.time()
        offsprings=[]
        for _ in range(N_pop):
            queue_out.put(binary_tournament(pop))
        while len(offsprings)!=N_pop:
            if not queue_in.empty():
                ofs=queue_in.get()
                if ofs and not inPop(ofs, pop):
                    offsprings.append(ofs)
                else:
                    queue_out.put(binary_tournament(pop))
            else:
                time.sleep(1e-3)
        pop+=offsprings
        print('in main', time.time() - b)
        pop=selection(pop)[:N_pop]
    for _ in range(len(process_pool)):
        queue_out.put(None)
    for p in process_pool:
        p.join()
    for p in process_pool:
        p.terminate()
    return pop

def master_slave_slave(queue_out:Queue, queue_in:Queue, args:dict,labels, mbs:np.ndarray, Pmbs:np.ndarray):
    crossover=gen_crossover_fun(args)
    mutation=gen_mutation_fun(args)
    P_m=args['P_m']
    cnt=0
    offspring=None
    while True:
        if not offspring:
            cnt+=1
        tmp=time.time()
        if not queue_in.empty():
            get=queue_in.get()
            if cnt%20==0:
                print("migration", time.time()-tmp)
            if not get:
                break
        else:
            time.sleep(0.1)
            continue
        offspring = crossover(get)
        offspring = mutation(offspring)
        offspring, rj_patterns, er_patterns = evaluate(offspring, labels, mbs, args)
        if offspring is None:
            queue_out.put(None)
            continue
        if random.random() < P_m:
            # consider they are the same
            mr_patterns = rj_patterns + er_patterns
            offspring = simf_gbml(offspring, mr_patterns, Pmbs, labels, args, mbs)
            if offspring is None:
                queue_out.put(None)
                continue
        queue_out.put(offspring)
    print(cnt)

def to_cf_q(rule:Rule, loc_data_info, label, loc_mbs:np.ndarray):
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
    cgs = []
    for xp_i in range(len(label)):
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
def evaluate(fuzzy_set_c: RuleSet, loc_label, loc_mbs:np.ndarray, args:dict)->(RuleSet,list,list):
    """
    fuzzy_set_c S, a fuzzy set, [ruleset, fitness, totalcorrect, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]  # win = cgs * CFq
    对于新生成的每个rule set 我们都补充完整它的信息，比如correct, Cq, Cfq等
    """
    reject_i = []
    error_i = []
    for rule in fuzzy_set_c:
        cgs, c_q, cf_q = to_cf_q(rule, args, loc_label, loc_mbs)
        rule.cgs_mul_cfq = [i * cf_q for i in cgs]  # win
        rule.correct=0
        rule.cq = c_q
        rule.cfq = cf_q
    for xp_i in range(len(loc_label)):
        win_rule(fuzzy_set_c, xp_i, reject_i, error_i, loc_label)
    # drop the meaningless fuzzy set
    retain_set = []
    for rule_i in range(len(fuzzy_set_c)):
        if not (fuzzy_set_c[rule_i].correct <= 0 or fuzzy_set_c[rule_i].cfq <= 0):  # CFq <= 0
            retain_set.append(rule_i)
        # else:
        #     if random.random() < 0.5:
        #         retain_set.append(rule_i)
    rule_set = [fuzzy_set_c[i] for i in retain_set]
    if len(rule_set) == 0:
        return None, None, None
    fuzzy_set_c.rules = rule_set
    fuzzy_set_c.objective['correct'] = sum([rule.correct for rule in rule_set])
    fuzzy_set_c.objective['rule_len'] = len(rule_set)
    return fuzzy_set_c, reject_i, error_i
def p_gen_ruleset(mr_x_i:list, loc_Pmbs:np.ndarray)->RuleSet:
    """
    mr_x_i: 根据这些训练数据根据公式14生成新的rule
    :param mr_x_i:
    :param loc_Pmbs:
    :return:
    """
    # 这里是为了生成rule set
    rules = []
    for i in mr_x_i:
        ante = []
        for j in range(loc_Pmbs.shape[1]):
            r = random.random()
            p = loc_Pmbs[i][j]
            for k in range(len(p)):
                if p[k] > r:
                    ante.append(k)
                    break
        rules.append(Rule(ante))
    return RuleSet(rules)
def pg_rule_set(loc_pop_size, loc_label, loc_mbs:np.ndarray, loc_n_rule, loc_Pmbs:np.ndarray, args:dict):
    """
    依概率（公式14生成种群）
    """
    pop = []
    while len(pop) != loc_pop_size:
        x_i = random.sample(range(len(loc_label)), loc_n_rule)
        rule_set_c = p_gen_ruleset(x_i, loc_Pmbs)
        rule_set_c, _, _ = evaluate(rule_set_c, loc_label, loc_mbs, args)
        if rule_set_c is None:
            continue
        pop.append(rule_set_c)
    return pop
# --------------------------------以下代码是测试集所需--------------------
#########################################################################
# 这个值其实是固定的，但我们不预先计算是因为时间复杂度太大太大，这也是为什么我们采用遗传来演化
# 其中 时间复杂度为 O( (nm)^len(fuzzy set) ), 其中 n是数据的维度， m是数据量
def tst_cg(rule:Rule, xp_i, tst_mbs):
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
def tst_to_cf_q(rule:Rule, tst_mbs, tst_dat):
    """
    rule_c:= a complete rule [[antecedent parts],cgs, Cq, CFq]
    """
    cgs = []
    for xp_i in range(tst_dat.shape[0]):
        cgs.append(tst_cg(rule, xp_i, tst_mbs))
    return cgs
def tst_win_rule(rules:RuleSet, xp_i, reject, error, tst_label):
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
        win_weights.append((rule.cgs_mul_cfq[xp_i], rule_i))
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
            if rules[fir[1]].cq == rules[sec[1]].cq:
                # same class
                if rules[fir[1]].cq == tst_label[xp_i]:
                    rules[fir[1]].correct.append(xp_i)
                else:
                    error.append(xp_i)
            else:
                # difference class, considered as rejected
                reject.append(xp_i)
        else:  # fir weight only the biggest
            if rules[fir[1]].cq == tst_label[xp_i]:
                rules[fir[1]].correct.append(xp_i)
            else:
                error.append(xp_i)
    else:
        # a valid rule
        if rules[fir[1]].cq == tst_label[xp_i]:
            rules[fir[1]].correct.append(xp_i)
        else:
            error.append(xp_i)
def tst_evaluate(rule_set, tst_mbs, tst_dat, tst_label):
    """
    fuzzy_set_c S, a fuzzy set, [ruleset, fitness, totalcorrect, ruleset_len]
    ruleset := [rules]
    rule := [[antecedent parts],cgs * CFq, Cq, CFq, correct]  # win = cgs * CFq ,对于每一个数据,它的win
    """
    reject_i = []
    error_i = []
    for rule in rule_set:
        cgs = tst_to_cf_q(rule, tst_mbs, tst_dat)
        rule.cgs_mul_cfq = [i * rule.cfq for i in cgs]  # win
        rule.correct = []  # store those patterns that been correct
    for xp_i in range(tst_dat.shape[0]):
        tst_win_rule(rule_set, xp_i, reject_i, error_i, tst_label)
    rule_set.objective['correct'] = sum([len(rule.correct) for rule in rule_set])
    rule_set.objective['rule_len'] = len(rule_set)
    return rule_set, reject_i, error_i
def _train_mode(mode, _data, _label):
    if mode == '5CV':
        tmp = int(_data.shape[0] * 0.8)
    elif mode == '10CV':
        tmp = int(_data.shape[0] * 0.9)
    else:
        tmp = _data.shape[0] // 2
    return _data[:tmp], _data[tmp:], _label[:tmp], _label[tmp:]



if __name__ == '__main__':
    begin_time=time.time()
    TODO = None
    run_which={
            'migration': RING,# 协作方式
            'objective': NSGA_II,
            'data_division': NODIVISION,
            'train-data':'sat.trn',
            'test-data':None,
            'evaluation_num': 1e10,
            'core_num': 6,
            'init':None,
            'init_N_pop':100,
            # 'crossover':TODO,
            # 'mutation':TODO,
            # 'selection':TODO,
            'stop_generation': 0,
            'N_pop':100,
            'weight-vector':(-1, 1),
            'P_M':0.1,
            'P_m':0.1,
            'lnum':0,
            'P_dc':0.8,
            'N_rule':10,
            'delimiter':' ',
            'mode':'5CV',
            'migration_fre':5,
            'migration_num':3,
            'copy_migration':True,
    }

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], "f:n:")
    for op, val in opts:
        for op, val in opts:
            if op == '-f':
                run_which['migration_fre']=int(val)
            elif op == '-n':
                run_which['migration_num']=int(val)
            elif op == '-c':
                run_which['core_num'] = int(val)
            elif op == '-t':
                run_times=int(val)
            elif op == '-m':
                if val=='True':
                    run_which['copy_migration']=True
                else:
                    run_which['copy_migration']=False
            elif op=='-r':
                run_which['train-data']=val
            elif op=='-s':
                run_which['test-data']=val
            elif op == '-d':
                run_which['delimiter'] = val


    # parallelism=\
    #     [
    #         'master-slave',
    #         'weight-vector',
    #         'modea-d',
    #         'nsga-ii'
    #     ]
    # dataset=\
    #     [
    #         'pima_.dat',
    #         'breast_w.csv',
    #         'gesture.csv',
    #         'glass.csv',
    #         'sonar.csv',
    #         'wine.csv'
    #     ]
    """ read data """
    train_data, tst_data, label, tst_label, label_dict = _preprecess_data(run_which)
    print('read data finish')
    print(label_dict)
    worker = getWorker(run_which)
    pop = worker(run_which, train_data, label)
    """ begin testing """
    g_fuzzy_set=_gen_fuzzy_set()
    tst_mbs = to_membership(tst_data, g_fuzzy_set)
    for i in pop:
        train_correct=i.objective['correct'][0]
        tst_evaluate(i, tst_mbs, tst_data, tst_label)
        print(i.objective['rule_len'][0],',',
              1-train_correct/len(label),',',
              1-i.objective['correct'][0]/len(tst_label)
              )
    print('finish training at:', time.time()-begin_time)
