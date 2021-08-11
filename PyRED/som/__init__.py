#from som_python import train_som, compute_u_matrix
from som_cython import train_som, compute_u_matrix

class SOM:
    
    def __init__(self, width, height, learning_rate=0.5, radius=0.25, \
        max_iter=100, random_seed=None, randomise_order=False, \
        neighbourhood_function="gauss"):
        
        self._width = width
        self._height = height
        self._learning_rate = learning_rate
        self._radius = radius
        self._max_iter = max_iter
        self._random_seed = random_seed
        self._randomise_order = randomise_order
        self._neighbourhood_function = neighbourhood_function
    
    def fit(self, X):
        
        self.w, self.e = train_som(X, self._width, self._height, \
            self._learning_rate, self._radius, self._max_iter, \
            self._random_seed, self._randomise_order, \
            self._neighbourhood_function)
        
        self.u = compute_u_matrix(self.w)


if __name__ == "__main__":
    
    import time
    import numpy
    from matplotlib import pyplot
    
    numpy.random.seed(19)
    
    X = numpy.random.randn(500, 2)
    X[:X.shape[0]//2,0] += 2
    X[X.shape[0]//2:,0] -= 2
    X[:X.shape[0]//4,1] -= 5
    
    pyplot.figure()
    pyplot.plot(X[:,0], X[:,1], 'o')
    pyplot.savefig("test.png")
    
    t0 = time.time()
    som = SOM(10, 8, random_seed=19, max_iter=100, neighbourhood_function="gauss")
    som.fit(X)
    t1 = time.time()
    
    print("SOM training duration: %.2f seconds" % (t1-t0))
    
    for i in range(som.w.shape[2]):
        pyplot.figure()
        pyplot.imshow(som.w[:,:,i])
        pyplot.savefig("test_w%d.png" % (i))

    pyplot.figure()
    pyplot.plot(som.e)
    pyplot.savefig("test_e.png")

    pyplot.figure()
    pyplot.imshow(som.u, cmap="gray_r")
    pyplot.savefig("test_u.png")

    pyplot.close("all")
    
    import somoclu
    t0 = time.time()
    som = somoclu.Somoclu(10,8)
    som.train(X)
    t1 = time.time()
    
    print("Somoclu SOM training duration: %.2f seconds" % (t1-t0))
    
    for i in range(som.codebook.shape[2]):
        pyplot.figure()
        pyplot.imshow(som.codebook[:,:,i])
        pyplot.savefig("test2_w%d.png" % (i))

    pyplot.figure()
    pyplot.imshow(som.umatrix, cmap="gray_r")
    pyplot.savefig("test2_u.png")

    pyplot.close("all")
    
