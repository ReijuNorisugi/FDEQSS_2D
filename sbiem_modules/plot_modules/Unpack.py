import struct
import numpy as np


def unpack(fname, length, num):
    param = np.zeros((length, num), dtype=np.float64) 
    f = open(fname, mode='rb')
    fmt = '<' + str(num) + 'd'
    nb = struct.calcsize(fmt)
    for i in range(length):
        bu = f.read(nb)
        data = struct.unpack(fmt, bu)
        param[i, :] = np.float64(data[:])
    return param