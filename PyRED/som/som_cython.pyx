from cpython cimport bool

import numpy
cimport numpy

def train_som(numpy.ndarray X, int width, int height, float learning_rate=0.5, \
        float radius=0.25, int max_iter=100, int random_seed=-1, \
        bool randomise_order=False, str neighbourhood_function="dist"):
    
    """
    """
    
    assert X.dtype == numpy.float64
    
    # Copy the SOM settings.
    cdef (int,int) dim = (height,width)
    cdef float _r = radius * max(width,height)
    cdef float _lr = learning_rate
    
    # Set the random seed.
    if random_seed >= 0:
        numpy.random.seed(random_seed)
    
    # Count the number of features and observations in this dataset.
    cdef int n_observations = X.shape[0]
    cdef int n_features = X.shape[1]
    
    # Set the SOM's depth (length of each weight vector).
    cdef int n_w = n_features
    
    # Initialise a network with random values between 0 and 1.
    cdef numpy.ndarray[numpy.float64_t, ndim=3] w = numpy.random.rand(dim[0], dim[1], n_w)
    # Transform the values to the range [-1, 1].
    w = (w * 2.0) -1.0
    
    # Determine the iteration order.
    cdef numpy.ndarray[numpy.int64_t, ndim=1] order = numpy.arange(0, n_observations, 1, dtype=numpy.int64)
    if randomise_order:
        numpy.random.shuffle(order)

    # Create a meshgrid to use for later node-to-node distance computing.
    cdef numpy.ndarray[numpy.float64_t, ndim=2] x
    cdef numpy.ndarray[numpy.float64_t, ndim=2] y
    cdef numpy.ndarray[numpy.float64_t, ndim=1] xi = numpy.arange(float(dim[1]))
    cdef numpy.ndarray[numpy.float64_t, ndim=1] yi = numpy.arange(float(dim[0]))
    x, y = numpy.meshgrid(xi, yi)
    
    # Static typing for iterators.
    cdef int s, i, j
    # Static typing for floats.
    cdef float f, r, lr, _max
    # Static typing for NumPy arrays.
    cdef numpy.ndarray[numpy.float64_t, ndim=3] w_prev
    cdef numpy.ndarray[numpy.float64_t, ndim=3] Di
    cdef numpy.ndarray[numpy.float64_t, ndim=2] d
    cdef numpy.ndarray[numpy.float64_t, ndim=2] nd
    cdef numpy.ndarray[numpy.float64_t, ndim=2] mask
    cdef (float,float) ci

    # Create a vector to hold change between iterations.
    cdef numpy.ndarray[numpy.float64_t, ndim=1] e = numpy.zeros(max_iter, dtype=numpy.float64)
    
    # Loop through all iterations.
    for s in range(max_iter):

        # Store the previous weights matrix.
        w_prev = numpy.copy(w)

        # Set the current learning rate and radius. These are downweighed
        # linearly throughout iterations.
        f = 1.0 - (float(s) / float(max_iter))
        r = _r * f
        lr = _lr * f
        
        # Loop through all input vectors.
        for i in range(n_observations):
            
            # Get the current input vector.
            Di = X[order[i],:].reshape((1,1,n_w))

            # Compute the distance between the current input vector and all
            # the node weights.
            d = numpy.sqrt(numpy.sum((w - Di)**2, axis=2))
            # Find the closest node (best matching unit).
            ci = numpy.unravel_index(numpy.argmin(d), (height,width))

            # Compute the distance between the best fitting node and all
            # other nodes.
            if neighbourhood_function in ["gauss", "cut_gauss"]:
                nd = (x-ci[0])**2 + (y-ci[1])**2
            else:
                nd = numpy.sqrt((x-ci[0])**2 + (y-ci[1])**2)
            
            # Linear distance neighbourhood function.
            # Constructs a mask that linearly decreases from the best
            # fitting unit to the radius.
            if neighbourhood_function == "dist":
                # Invert the node distances. Higher values now indicate nodes
                # that are closer to the best matching unit.
                _max = numpy.max(nd)
                nd = (_max - nd)
                # Scale the values according to the current radius.
                nd -= (_max - r)
                # Rescale the node distance weights to the range [-inf,1].
                nd = nd / r
                # Create a mask that is 0 for all negative values, and 1 for
                # all positive values.
                mask = (nd>=0).astype(numpy.float64)
                # Multiply with the mask to set all negative values to 0.
                nd = nd * mask

            # Bubble neighbourhood function.
            # Constructs a mask that is 1 for all units between the best
            # fitting unit and the radius, and 0 everywhere else.
            elif neighbourhood_function == "bubble":
                # Only select the distances under the radius.
                nd = (nd < r).astype(numpy.float64)
            
            # Gaussian neighbourhood function.
            # Constructs a mask that is 1 for the best matching unit, and
            # decreases as per a Gaussian with sigma=radius.
            elif neighbourhood_function == "gauss":
                # Down-weigh distances according to a Gaussian around the
                # best fitting unit.
                nd = numpy.exp(-1.0 * (nd / (2.0*(r**2.0))))

            # Gaussian neighbourhood function.
            # Constructs a mask that is 1 for the best matching unit, and
            # decreases as per a Gaussian with sigma=radius.
            elif neighbourhood_function == "cut_gauss":
                # Create a mask to filter out all distances above the
                # current radius.
                mask = (numpy.sqrt(nd) > r).astype(numpy.float64)
                # Down-weigh distances according to a Gaussian around the
                # best fitting unit.
                nd = numpy.exp(-1.0 * (nd / (2.0*(r**2.0))))
                # Cut out any values beyond the current radius.
                nd = nd * mask

            # Loop through all featurs.
            for j in range(n_features):
                # Update the node matrix.
                w[:,:,j] += nd * lr * (Di[0,0,j] - w[:,:,j])
        
        # Compute the difference between the current and the previous weights
        # matrix.
        e[s] = numpy.sum(numpy.sqrt(numpy.sum(w - w_prev, axis=2)**2))
    
    # Return the weights
    return w, e
        

def compute_u_matrix(numpy.ndarray w):
    
    assert w.dtype == numpy.float64
    
    # Static typing for iterators.
    cdef int row, col
    # Static typing for floats.
    cdef float m
    # Static typing for NumPy arrays.
    cdef numpy.ndarray[numpy.float64_t, ndim=2] u
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dx
    cdef numpy.ndarray[numpy.float64_t, ndim=2] dy
    cdef numpy.ndarray[numpy.int64_t, ndim=2] x
    cdef numpy.ndarray[numpy.int64_t, ndim=2] y
    cdef numpy.ndarray[numpy.int64_t, ndim=1] i_row
    cdef numpy.ndarray[numpy.int64_t, ndim=1] i_col
    cdef numpy.ndarray[numpy.int64_t, ndim=1] i_even_rows
    cdef numpy.ndarray[numpy.int64_t, ndim=1] i_even_cols
    cdef numpy.ndarray[numpy.int64_t, ndim=1] i_odd_rows
    cdef numpy.ndarray[numpy.int64_t, ndim=1] i_odd_cols
    cdef (int,int) shape

    # Compute the U matrix. This is a slightly odd concept, but in essence
    # it defines the distances between each of the nodes. The distance
    # between node A and node B is the Euclidean distance between the
    # weights of node A and node B. These distances only exist between
    # nodes, but the U matrix has more than just those. In-between nodes
    # are filled with the average of the distance nodes.
    # First, construct an empty matrix to hold the distance values in.
    shape = (w.shape[0] * 2 - 1, w.shape[1] * 2 - 1)
    u = numpy.zeros(shape, dtype=numpy.float64)
    # Compute the distance between horizontal neighbours.
    dx = numpy.sum(numpy.diff(w, axis=0)**2, axis=2)
    # Compute the distance between vertical neighbours.
    dy = numpy.sum(numpy.diff(w, axis=1)**2, axis=2)
    # Store the horizontal distances in the odd x indices, and the even
    # y indices.
    i_col = numpy.arange(1, shape[0], 2, dtype=numpy.int64)
    i_row = numpy.arange(0, shape[1], 2, dtype=numpy.int64)
    y, x = numpy.meshgrid(i_row, i_col)
    u[x,y] = dx
    # Store the vertical distances in the even x indices, and the odd
    # x indices.
    i_col = numpy.arange(0, shape[0], 2, dtype=numpy.int64)
    i_row = numpy.arange(1, shape[1], 2, dtype=numpy.int64)
    y, x = numpy.meshgrid(i_row, i_col)
    u[x,y] = dy
    # Go through all the spaces directly between the ones we just computed,
    # and make them the average of all surrounding spaces.
    i_even_cols = numpy.arange(0, shape[1], 2, dtype=numpy.int64)
    i_even_rows = numpy.arange(0, shape[0], 2, dtype=numpy.int64)
    for row in i_even_rows:
        for col in i_even_cols:
            # Top-left corner.
            if row == 0 and col == 0:
                m = (u[0,1] + u[1,0]) / 2.0
            # Top-right corner.
            elif row == 0 and col == i_even_cols[-1]:
                m = (u[0,i_even_cols[-1]-1] + u[1,i_even_cols[-1]]) / 2.0
            # Bottom-left corner.
            elif row == i_even_rows[-1] and col == 0:
                m = (u[i_even_rows[-1]-1,0] + u[i_even_rows[-1],1]) / 2.0
            # Bottom-right corner.
            elif row == i_even_rows[-1] and col == i_even_cols[-1]:
                m = (u[i_even_rows[-1]-1,i_even_cols[-1]] + u[i_even_rows[-1],i_even_cols[-1]-1]) / 2.0
            # Top row.
            elif row == 0:
                m = (u[row,col-1] + u[row,col+1] + u[row+1,col]) / 3.0
            # Bottom row.
            elif row == i_even_rows[-1]:
                m = (u[row,col-1] + u[row,col+1] + u[row-1,col]) / 3.0
            # Left-most column.
            elif col == 0:
                m = (u[row-1, col] + u[row+1, col] + u[row, col+1]) / 3.0
            # Right-most column.
            elif col == i_even_cols[-1]:
                m = (u[row-1, col] + u[row+1, col] + u[row, col-1]) / 3.0
            # All other cells.
            else:
                m = (u[row-1,col] + u[row+1,col] + u[row,col-1] + u[row,col+1]) / 4.0
            
            # Add the computed average to the U matrix.
            u[row,col] = m

    # Go through all the spaces directly between the averages we just
    #  computed, and make them the average of all surrounding spaces.
    i_odd_cols = numpy.arange(1, shape[1], 2, dtype=numpy.int64)
    i_odd_rows = numpy.arange(1, shape[0], 2, dtype=numpy.int64)
    for row in i_odd_rows:
        for col in i_odd_cols:
            # Add the computed average to the U matrix.
            m = (u[row-1,col] + u[row+1,col] + u[row,col-1] + u[row,col+1]) / 4.0
            u[row,col] = m
    
    # Return the U matrix.
    return u
