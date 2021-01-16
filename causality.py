import numpy as np
import matplotlib.pyplot as plt

MATRIX_SIZE = 3
N_SAMPLES = 1024

class World:
    def __init__(self, size):
        #self.mat = np.random.randint(low=0, high=6, size=[size,size])
        #self.mat = (self.mat<2)
        #self.mat = self.mat.astype(np.int64)
        #make a directed acyclic graph by taking the lower triangle
        #self.mat = np.tril(self.mat, k=-1) 
        
        self.mat = np.zeros(shape=[size,size])
        self.mat[2,0] = 1
        self.mat[2,1] = 1
        
        self.shuffle = np.arange(0,size)
        #np.random.shuffle(self.shuffle)
        
    def rand(self):
        return np.random.normal(size=[])
        
    def run(self):
        result = np.zeros(shape=[self.mat.shape[0]])
        for c in range(self.mat.shape[0]):
            sum = np.sum(self.mat[c])
            if sum == 0:
                result[c] = self.rand()
            else:
                result[c] = 0
                for c_prev in range(0,c):
                    if self.mat[c,c_prev] == 1:
                        result[c] += result[c_prev]
        return result[self.shuffle]

class Agent:
    def __init__(self):
        pass

def fit(X, Y):
    #print(f"Fitting: {X} {Y}")
    N = X.shape[0]
    
    gen_mid    =  (np.dot(Y,X)-np.sum(Y)*np.sum(X)/N)
    gen_from_x = -(np.dot(X,X)-np.sum(X)*np.sum(X)/N)
    gen_from_y = -(np.dot(Y,Y)-np.sum(Y)*np.sum(Y)/N)
    
    print(f"dots {np.dot(X,X)} {np.dot(X,Y)} {np.dot(Y,Y)}")
    print(f"sums {np.sum(X)*np.sum(X)/N} {np.sum(X)*np.sum(Y)/N} {np.sum(Y)*np.sum(Y)/N}")
    print(f"divs {np.sum(X)*np.sum(X)/N/np.dot(X,X)} {np.sum(X)*np.sum(Y)/N/np.dot(X,Y)} {np.sum(Y)*np.sum(Y)/N/np.dot(Y,Y)}")
    
    
    if gen_from_x*gen_from_x > gen_from_y*gen_from_y:
        gen_x = gen_mid
        gen_y = gen_from_x
    else:
        gen_x = gen_from_y
        gen_y = gen_mid
        
    gen_b = (sum(Y)*gen_y+sum(X)*gen_x)/N
    print(f"chos {gen_x}x + {gen_y}y + {gen_b}=0")
    return

def fit_perpendicular(X,Y):
    N = X.shape[0]

    a = (np.dot(Y,Y)-np.sum(Y)*np.sum(Y)/N)
    b = (np.dot(X,Y)-np.sum(X)*np.sum(Y)/N)
    c = (np.dot(X,X)-np.sum(X)*np.sum(X)/N)
    t = np.arctan2(-2.0*b,a-c)*0.5
    bias = (np.cos(t)*np.sum(X) + np.sin(t)*np.sum(Y))/N
    print(np.degrees(t), bias)
    
    val_y = np.linspace(-5,5,100)
    val_x = np.full(shape=val_y.shape, fill_value=bias)
    coords = np.stack([val_x,val_y], axis=-1)
    rot_matrix = np.array([[np.cos(t),np.sin(t)],[-np.sin(t),np.cos(t)]])
    coords = np.matmul(coords,rot_matrix)

    plt.plot(X,Y,'bo')
    plt.plot(coords[:,0],coords[:,1],'-r')
    plt.show()

    
def fitboth(xs,ys):
    fit(xs,ys)
    fit(ys,xs)

#fit_perpendicular(np.array([1.47,1.50,1.52,1.55,1.57,1.60,1.63,1.65,1.68,1.70,1.73,1.75,1.78,1.80,1.83]), np.array([52.21,53.12,54.48,55.84,57.20,58.57,59.93,61.29,63.11,64.47,66.28,68.10,69.92,72.19,74.46]))

#fit(np.array([0,0,0,0,0]),np.array([1,2,3,4,5]))
#fitboth(np.array([1,1,1,1,1]),np.array([1,2,3,4,5]))

#fitboth(np.array([1,2,3]),np.array([1,2,4]))

#fit(np.random.normal(size=[2**20]),np.random.normal(size=[2**20]))

#fit_perpendicular(np.array([1,2,3]),np.array([2,4,6]))
#fit_perpendicular(np.array([2,4,6]),np.array([1,2,3]))
#fit_perpendicular(np.array([1,2]),np.array([2,4]))
#fit_perpendicular(np.array([0,1,2,3,4]),np.array([2,1,0,-1,-2]))
#fit_perpendicular(np.array([0,1,2,3,4]),np.array([-2,-1,0,1,2]))

#fit_perpendicular(np.array([0.998,0.999,1.000,1.001,1.002]),np.array([0,1,2,3,4]))

#fit_perpendicular(np.array([3,3,3,3,3]),np.array([1,2,3,4,5]))
#fit_perpendicular(np.array([1,3,5,2,4]),np.array([3,3,3,3,3]))

fit_perpendicular(np.array([-1,-2,-3,-4,-5]),np.array([1,2,3,4,5]))

exit(0)
    
        
world = World(MATRIX_SIZE)
print(world.mat)

data = np.zeros(shape=[N_SAMPLES,MATRIX_SIZE])
for sample_id in range(N_SAMPLES):
    result = world.run()
    data[sample_id] = result

xs = data[:,0]    
ys = data[:,2]

fit(xs,ys)


plt.plot(xs,ys,'bo')
plt.show()


"""
Ax + By + C = 0
-By = Ax + C
y = (Ax+C)/-B
y = -(Ax+C)/B
y = -Ax/B - C/B
"""