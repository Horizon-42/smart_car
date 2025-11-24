import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time

#Data
x = np.array([8.0]) #pixel value
y = np.array([5.0]) #throttle value

#Model
def model(x,w): return w.T*x 

#Initial prediction
w0 = np.array([-1.5])
y_hat = model(x,w0)
print("Predicted: " + str(y_hat))
print("Truth: " + str(y))
input()

#Loss function
def loss(y_pred, y_truth): return (y_pred[:,None]-y_truth[:,None])**2

#Visual solution
if False:
    w_candidates = np.linspace(-2,2,1000)
    plt.plot(w_candidates,loss(model(x,w_candidates),y),linewidth=4)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('w', fontsize=20)
    plt.ylabel('L(w)', fontsize=20)
    plt.show()

#Algorithmic solution
alpha = 1e-4
epsilon = 1e-2
wt = w0

lossGradient = 2*x*(wt*x-y)
while norm(lossGradient)>epsilon:
    wt = wt-alpha*lossGradient
    lossGradient = 2*x*(wt*x-y)
    print('Prediction: ' + str(np.round(model(x,wt),3)) + "   ||   Loss: " + str(np.round((model(x,wt)-y)**2,3))  + "   ||   Loss gradient: " + str(np.round(norm(lossGradient),3)) + "   ||   w: " + str(np.round(wt,3)))
    time.sleep(0.1)