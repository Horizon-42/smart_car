import numpy as np
import time
import pdb

#data
signal_size = int(1e5)
kernel_size = int(1e2)
x = np.random.rand(signal_size) 
k = np.random.rand(2*kernel_size+1)

#numpy
start_numpy = time.time()
xk_numpy = np.convolve(x,k,mode='valid')
end_numpy = time.time()
print('Runtime (numpy):' + str(end_numpy-start_numpy))

#naive
start_naive = time.time()
xk_naive = np.zeros(len(x))
for i in range(kernel_size,len(x)-kernel_size):
    sum = 0
    for j in range(len(k)):
        sum += x[i+kernel_size-j]*k[j]
    xk_naive[i] = sum
end_naive = time.time()
print('Runtime (naive):' + str(end_naive-start_naive))

#fft
start_fft = time.time()
N = (2**np.ceil(np.log2(len(x)+len(k)-1))).astype(int)
x_ext = np.hstack((x,np.zeros(N-len(x))))
k_ext = np.hstack((k,np.zeros(N-len(k))))
xk_fft = np.fft.ifft(np.fft.fft(x_ext)*np.fft.fft(k_ext)).real
end_fft = time.time()
print('Runtime (fft):' + str(end_fft-start_fft))

print(np.all(np.isclose(xk_numpy,xk_naive[kernel_size:len(x)-kernel_size])))
print(np.all(np.isclose(xk_numpy,xk_fft[2*kernel_size:len(x)])))












