import numpy as np

def computeBilinerWeights(q):

    ## TODO 2.1
    ## - Compute bilinear weights for point q
    ## - Entry 0 weight for pixel (x, y)
    ## - Entry 1 weight for pixel (x + 1, y)
    ## - Entry 2 weight for pixel (x, y + 1)
    ## - Entry 3 weight for pixel (x + 1, y + 1)

    x,y = q
    a = x - np.floor(x) 
    b = y - np.floor(y)
    

    w0 = (1-a)*(1-b)
    w1 = a*(1-b)
    w2 = b*(1-a)
    w3 = a*b
     

    weights = [w0, w1, w2, w3]

    return weights

def computeGaussianWeights(winsize, sigma):

    ## TODO 2.2
    ## - Fill matrix with gaussian weights
    ## - Note, the center is ((winSize.width - 1) / 2,winSize.height - 1) / 2)

    #weights = [1, 0, 0, 0]

    
    height, width = winsize
    # Create a meshgrid for the window
    y, x = np.mgrid[0:height, 0:width]
    
    # Center of the window
    c_x = (width - 1) / 2
    c_y = (height - 1) / 2
    
    # Compute centered and normalized coordinates
    bar_x = (c_x - x) / width
    bar_y = (c_y - y) / height
    
    # Compute Gaussian weights
    weights = np.exp(-(bar_x**2 + bar_y**2) / (2 * sigma**2))
    
    return weights

def invertMatrix2x2(A):

    ## TODO 2.3
    ## - Compute the inverse of the 2 x 2 Matrix A
    
    #invA = np.identity(2)

    invA = np.linalg.inv(A)

    return invA
