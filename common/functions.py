# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


#np.array([0,1,2]) 
# ndim:1, shape:(3,)
#np.array([[0,1,2],[3,4,5]])
# ndim:2, shape:(2,3)
#np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]],[[12,13,14],[15,16,17]]])
# ndim:3, shape:(4,2,3)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        #np.array([0,1,2]) -> np.array([[0,1,2]])
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    # axis0 : sample index, axis1 : value index
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    # np.arange(100) -> [0, 1, 2, 3, .. , 99]
    # -np.sum(t * np.log(y))
    #
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    # [before optimization...]
    # sum = 0
    # for idx in range(0, batch_size):
    #    sum = -np.sum(t[idx] * np.log(y[idx] + 1e-7))
    # return sum


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
