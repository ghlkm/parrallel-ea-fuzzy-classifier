import numpy as np
xte=np.genfromtxt(fname='final_X_test.txt', delimiter=',')
xtr=np.genfromtxt(fname='final_X_train.txt', delimiter=',')
yte=np.genfromtxt(fname='final_y_test.txt', delimiter=',')
ytr=np.genfromtxt(fname='final_y_train.txt', delimiter=',')
yte=np.reshape(yte, (len(yte), 1))
ytr=np.reshape(ytr, (len(ytr), 1))
te=np.hstack([xte, yte])
tr=np.hstack([xtr, ytr])
s=''
for i in range(xte.shape[1]):
   s+='%.5f, '
s+='%d'
np.savetxt('activity_test.csv', te, delimiter=',', fmt=s)
np.savetxt('activity_train.csv', tr,  delimiter=',', fmt=s)