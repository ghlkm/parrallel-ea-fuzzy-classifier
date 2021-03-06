#! /bin/bash
fre=(1 5 10 20)
num=(1 5 10 20)
core_num=6
train_data='Breast_w.csv'
test_data=''
delimiter=','
copy=('True' 'False')

run_times=2
for c in ${copy[@]};do
    for f in ${fre[@]};do
	    for n in ${num[@]};do
	        echo '#!/bin/bash' > mypbs.pbs
            echo '#PBS -N task_2019_6_3' >> mypbs.pbs
            echo '#PBS -l nodes=1:ppn='$core_num >> mypbs.pbs
            echo '#PBS -q fat' >> mypbs.pbs
            echo '#PBS -V' >> mypbs.pbs
            echo '#PBS -S /bin/bash' >> mypbs.pbs

            echo 'EXEC=/home/cs-likm/dataset/GAs.py' >> mypbs.pbs
            echo 'cd $PBS_O_WORKDIR' >> mypbs.pbs
            echo 'export OMP_NUM_THREADS='$core_num >> mypbs.pbs
            echo 'export PATH=/opt/software/python-3.6.3/bin:$PATH' >> mypbs.pbs
            echo 'python3 $EXEC'  -f $f -n $n -c $core_num -t $run_times -m $c -r $train_data -s $test_data -d $delimiter >> mypbs.pbs
            echo >> mypbs.pbs
            qsub mypbs.pbs
	    done
    done
done