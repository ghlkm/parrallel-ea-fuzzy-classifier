题目： 并行的演化机器学习算法比较
由于机器的充足（数量多但是每个机器的性能有限）， 当前的计算性能比较适合多机器协作运行算法， 比如我们组的实验器材就十分充足而且学校上有超算
当前提出了很多的并行演化策略，但是没有人正式地做过与他人的对比， 亟需有人能给出一定的比较结果，进一步研究并行演化算法在某些情况下的表现

控制实验条件，用固定的模糊分类器， 探究相关参数或算法的并行效率， 这个效率包括运行时间，结果的好坏
总的来说这是一个比较实验

# 要注意用不同的分类数据， 比如bias， 维度， 数据量 
# 有时间应该运行多次
# 可能会用到一些检验方法，即某条件下， 某并行策略比另一个好(坏， 一样)

相关要研究的问题可能有: 
	CPU的核数
		核数在某些算法里是种群多少的概念， 有些则是速度的概念
		当前得到的一些结论是，在固定数据集的基础上，运行速度随CPU的核数变化 形状 形如一个 二次函数，
		也就是当核数增加时，运算速度会增加，但当增加太多时， 由于进程间通讯带来的副作用（如锁， 等待， 公共数据读写等），导致速度随核数变慢
		
	(1) What is the level of communication necessary to make a parallel GA behave like a panmictic GA? 
	(2) What is the cost of this communication? And, 
	(3) is the communication cost small enough to make this a viable alternative for the design of parallel GAs?
	上面三个问题是parallel.pdf的论文提出的几个问题， 这几个问题涉及的其实都是同一个问题， 如何migration?
	migration 会影响多个种群总体的多样性，会影响计算性能， 大家希望知道的是 交流需要多少时间， 怎么保证交流不会太多但是种群演化又能相互促进
	有一些研究者基于自己对migration的看法，认为migration is enabled until the population was close to converge
	migration 是一个值得探究的问题
	
	A very important theoretical question is to determine if (and under what conditions) a parallel GA can ﬁnd a better solution than a serial GA
	这个问题要求我们在同样evaluation的情况下，希望能得到 并行 与 非并行 的GA相比到底如何， 直接将演化任务分配给各个核好，还是各个核各自演化协作好，还是说应该处于中间
	这个可能需要有一个随着evaluation的变化，解如何变化的一个曲线，但是纵坐标应该是怎么样的？


	不同的并行策略其实就是data-division, selection, migration, data-migration的一个组合， 当前重点是在migration上
	
	
相关硬件：
	HPC: Hyper Performance Computer， 太乙


相关问题
	多目标优化：模糊分类器的目标数， 训练的精确度
    
	演化机器学习

	模糊分类器
		1+2+3+4+5的三角模糊分类器


当前进度
	目前的工作重心可能将重点放在探究migration上， 完整的算法目前综合在一个py上，有时间应该会把结构弄好一些
	parallel.pdf已经基本看完，这是一篇综述，大概讲的是 不接近现在这个时间(2010年前) 大概有哪些并行的策略，他们单独做过哪些比较有哪些结论
	其余几篇是可能准备看的比较新一点的几篇，有一个是一本书，可能只是会粗略看看