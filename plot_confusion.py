# run with python2

import numpy as np
import matplotlib.pyplot as plt

# conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], 
#             [3,31,0,0,0,0,0,0,0,0,0], 
#             [0,4,41,0,0,0,0,0,0,0,1], 
#             [0,1,0,30,0,6,0,0,0,0,1], 
#             [0,0,0,0,38,10,0,0,0,0,0], 
#             [0,0,0,3,1,39,0,0,0,0,4], 
#             [0,2,2,0,4,1,31,0,0,0,2],
#             [0,1,0,0,0,0,0,36,0,2,0], 
#             [0,0,0,0,0,0,1,5,37,5,1], 
#             [3,0,0,0,0,0,0,0,0,39,0], 
#             [0,0,0,0,0,0,0,0,0,0,38]]


conf_arr = [[7949,   43,   51,   91,  121,   89,   50,   21,   28,   31],
            [  57,   47,    1,    0,    2,    1,    1,    0,    0,    0],
            [  52,    0,   46,    2,    2,    0,    4,    0,    3,    0],
            [1063,   10,   21,  605,   17,   34,   11,    5,    8,    7],
            [1294,   20,   10,   50,  585,   37,    9,    4,    9,    0],
            [ 198,    4,    0,   11,   12,  188,    0,    1,    2,    0],
            [  23,    1,    1,    1,    2,    0,   34,    1,    1,    1],
            [  41,    1,    0,    0,    0,    2,    3,   29,    0,   29],
            [  26,    0,    0,    0,    4,    1,    0,    0,   27,    0],
            [  22,    0,    0,    0,    0,    0,    2,    5,    1,   35]]


norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

# width, height = conf_arr.shape
width, height = 10, 10

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
labels = ('N', 'B_lr','B_rl', 'C_lr','C_lr', 'O','BE_lr', 'BE_rl','EN_lr', 'EN_rl')
plt.xticks(range(width), labels)
plt.yticks(range(height), labels)
plt.savefig('confusion_matrix.png', format='png')
