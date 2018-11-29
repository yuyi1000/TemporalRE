
import numpy

def get_pos_embedding():
    pos_embed = numpy.zeros((1, 15), dtype=numpy.int32)
    for i in range(15):
        l = numpy.zeros((1, 15), dtype=numpy.int32)
        l[0][i] = 1
        pos_embed = numpy.concatenate((pos_embed, l))
    return pos_embed


pos = get_pos_embedding()
print (pos)
