import numpy as np
np.set_printoptions(precision=5,suppress=True)
import matplotlib.pyplot as plt

MATRIX_SIZE = 6
N_SAMPLES = 1024

def rand_normal():
    return np.random.normal(size=[])

def get_min_index(m):
    return np.unravel_index(np.argmin(m),m.shape)

#gets the coefficients for a number of variables, and a maximum degree of polynomial
def get_coefs(vars, degree):
    assert(vars > 0)
    if degree == 0:
        return np.array([[0]*vars])
    
    results = np.zeros(shape=[0,vars])
    
    degrees = np.arange(start=0,stop=degree+1)
    curr_degrees = np.expand_dims(degrees,axis=-1)
    
    for next_var in range(1,vars):
        curr_degrees = np.expand_dims(curr_degrees,axis=0)
        next_degrees = np.arange(start=0,stop=degree+1)[:,None]
        next_degrees = np.repeat(next_degrees,repeats=curr_degrees.shape[1],axis=-1)
        curr_degrees = np.repeat(curr_degrees,repeats=degree+1,axis=0)
        
        next_degrees = np.expand_dims(next_degrees,axis=-1)
        
        result = np.concatenate([curr_degrees, next_degrees],axis=-1)
        result = result.reshape([-1,result.shape[-1]])
        
        new_result = []
        for n in range(result.shape[0]):
            if np.sum(result[n]) <= degree:
                new_result.append(result[n])
        
        result = np.stack(new_result)
        curr_degrees = result
    return curr_degrees
    
coefs = get_coefs(vars=5,degree=2)


#fit a polynomial of degree "degree" to points [X,Y]
#using linear regression
#returns: 1. coefficients
#         2. residuals
def fit_polynomial_regression(X, Y, degree):
    vars = X.shape[-1]
    
    coefs = get_coefs(vars=vars, degree=degree)
    coefs = np.transpose(coefs)
    
    XD = np.repeat(X[:,:,None],coefs.shape[-1],axis=-1)
    
    result = np.power(XD,coefs)
    result = np.where(coefs>0, result, np.zeros_like(result))
    result = np.sum(result,axis=-2)
    
    coefsum = np.sum(coefs,axis=-2)
    result += np.where(coefsum==0,np.ones_like(coefsum),np.zeros_like(coefsum))
    XB = result
    XBT = np.transpose(XB)
    invd = np.linalg.inv(np.dot(XBT,XB))
    result = np.dot(Y,np.dot(XB,invd))
    residuals = np.square(np.dot(XB,result)-Y)
    return result,residuals


#fits a line to points [X,Y]
#this is *not* linear regression.
#it's some kind of symmetric version
#but i haven't quite figured out what.
#
#i derived it myself with help from discord
#
#returns: 1. the angle at which the resulting
#            line is in radians
#         2. the bias of the line before rotating it
#         3. residuals
#-Angela McEgo
def fit_general(X,Y):
    N = X.shape[0]

    a = (np.dot(Y,Y)-np.sum(Y)*np.sum(Y)/N)
    b = (np.dot(X,Y)-np.sum(X)*np.sum(Y)/N)
    c = (np.dot(X,X)-np.sum(X)*np.sum(X)/N)
    t = np.arctan2(-2.0*b,a-c)*0.5 #this is where the magic happens
    bias = (np.cos(t)*np.sum(X) + np.sin(t)*np.sum(Y))/N
    
    coords = np.stack([X,Y], axis=-1)
    rot_matrix = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    coords = np.matmul(coords,rot_matrix)
    residuals = np.square(coords[:,0]-bias)
    return t,bias,residuals

class World:
    def __init__(self, size):
        """self.mat = np.zeros(shape=[size,size], dtype=np.float64)
        
        self.mat[1,0] = 1
        self.mat[2,0] = 1
        self.mat[3,1] = 1
        self.mat[3,2] = -1"""
        
        """self.mat[1,0] = 2
        self.mat[2,1] = 3"""
        
        #make a random connectivity graph
        self.mat = np.random.randint(low=0, high=6, size=[size,size])
        self.mat = (self.mat<3)
        
        self.mat = np.where(self.mat,np.random.normal(size=[size,size]),np.zeros_like(self.mat))
        
        #make it into a directed acyclic graph by taking the lower triangle
        self.mat = np.tril(self.mat, k=-1) 
        
        self.shuffle = np.arange(0,size)
        #np.random.shuffle(self.shuffle)
        
    def run(self, set_values=None):
        result = np.zeros(shape=[self.mat.shape[0]])
        for c in range(self.mat.shape[0]):
            if set_values is not None and set_values[c] is not None:
                result[c] = set_values[c]
            else:
                sum = np.sum(np.square(self.mat[c]))
                if sum == 0:
                    result[c] = rand_normal()
                else:
                    result[c] = 0
                    for c_prev in range(c):
                        result[c] += self.mat[c,c_prev]*result[c_prev]
                    
        return result[self.shuffle]

class Agent:
    def __init__(self):
        pass

def test_values(x, y):
    data = np.zeros(shape=[N_SAMPLES,MATRIX_SIZE])
    for sample_id in range(N_SAMPLES):
        set_values = [None]*MATRIX_SIZE
        for xid in x:
            set_values[xid] = rand_normal()
        result = world.run(set_values=set_values)
        data[sample_id] = result

    #plt.plot(data[:,x],data[:,y],'bo')
    #plt.plot([-4,4],[-4,4],'-r')
    #plt.show()
    wb,res = fit_polynomial_regression(data[:,x],data[:,y],degree=1)
    
    w = wb[1:]
    b = wb[0]
    
    print(f"{x}->{y}: w={w} b={b} res={np.mean(res)}")
    return w,b,np.mean(res)
        

world = World(MATRIX_SIZE)

model = np.zeros(shape=[MATRIX_SIZE,MATRIX_SIZE])

while True:
    print("------------------------------")
    data = np.zeros(shape=[N_SAMPLES,MATRIX_SIZE])
    for sample_id in range(N_SAMPLES):
        result = world.run()
        data[sample_id] = result

    #residuals = 
    residuals = np.zeros(shape=[MATRIX_SIZE,MATRIX_SIZE])
    angles = np.zeros(shape=[MATRIX_SIZE,MATRIX_SIZE])
    for first in range(MATRIX_SIZE):
        others = []
        for second in range(MATRIX_SIZE):
            if first==second:
                continue
            others.append(second)
        others = np.array(others)
        w,b,r = test_values(others,first)
        model[first,others] = w
    break
    
print(model)
print("real mat")
print(world.mat)

