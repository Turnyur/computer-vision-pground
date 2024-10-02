import numpy as np
import cv2

from .util import *


class OpticalFlowLK:

    def __init__(self, winsize, epsilon, iterations):

        self.winsize = winsize
        self.epsilon = epsilon
        self.iterations = iterations

    def compute(self, prevImg, nextImg, prevPts):

        assert prevImg.size != 0 and nextImg.size != 0, "check prevImg and nextImg"
        assert prevImg.shape[0] == nextImg.shape[0], "size mismatch, rows."
        assert prevImg.shape[1] == nextImg.shape[1], "size mismatch, cols."

        N = prevPts.shape[0]
        status = np.ones(N, dtype=int)
        nextPts = np.copy(prevPts)

        prevDerivx = []
        prevDerivy = []
        
        ## TODO 2.4
        ## Compute the spacial derivatives of prev using the Scharr function
        ## - Make sure to use the normalized Scharr filter!!

        prevDerivx = cv2.Scharr(prevImg, cv2.CV_16S, 1,0)
        prevDerivy = cv2.Scharr(prevImg, cv2.CV_16S, 0,1)

        prevDerivx = cv2.convertScaleAbs(prevDerivx)  # Normalize
        prevDerivy = cv2.convertScaleAbs(prevDerivy)


        halfWin = np.array([(self.winsize[0] - 1) * 0.5, (self.winsize[1] - 1) * 0.5])
        weights = computeGaussianWeights(self.winsize, 0.3)

        for ptidx in range(N):
            u0 = prevPts[ptidx]
            u0 -= halfWin

            u = u0
            iu0 = [int(np.floor(u0[0])), int(np.floor(u0[1]))]

            if iu0[0] < 0 or \
                    iu0[0] + self.winsize[0] >= prevImg.shape[1] - 1 or \
                    iu0[1] < 0 or \
                    (iu0[1] + self.winsize[1] >= (prevImg.shape[0] - 1)):
                status[ptidx] = 0
                continue

            #bw = np.array(computeBilinerWeights(u0))
            bw = np.array(computeBilinerWeights(u0)).reshape(self.winsize[1], self.winsize[0])


            bprev = np.zeros((self.winsize[0] * self.winsize[1], 1))
            A = np.zeros((self.winsize[0] * self.winsize[1], 2))
            AtWA = np.zeros((2, 2))
            invAtWA = np.zeros((2, 2))

            for y in range(self.winsize[1]):
                for x in range(self.winsize[0]):
                    gx = int(iu0[0] + x)
                    gy = int(iu0[1] + y)

                    ## TODO 3.1
                    ## Compute the following parts of step 2.
                    ##   bprev      Size: (w*h) x 1 Matrix
                    ##   A          Size: (w*h) x 2 Matrix
                    ##   AtWA       Size:     2 x 2 Matrix
                    ## Use the bilinear weights bw!
                    ## W is stored in 'weights', but not as a diagonal matrix!!!

                    bprev[index, 0] = bw[y, x] * (prevImg[gy, gx] - nextImg[gy, gx])
                    A[index, 0] = bw[y, x] * prevDerivx[gy, gx]
                    A[index, 1] = bw[y, x] * prevDerivy[gy, gx]

                    index += 1

            AtWA = np.dot(A.T, A)
                    
                    



            ## TODO 3.1
            ## Compute invAtWA
            ## Use the function invertMatrix2x2
            invAtWA = invertMatrix2x2(AtWA)

            ## Estimate the target point with the previous point
            u = u0

            ## Iterative solver
            for j in range(self.iterations):
                iu = [int(np.floor(u[0])), int(np.floor(u[1]))]

                if iu[0] < 0 or iu[0] + self.winsize[0] >= prevImg.shape[1] - 1 \
                        or iu[1] < 0 or iu[1] + self.winsize[1] >= prevImg.shape[0] - 1:
                    status[ptidx] = 0
                    break

                bw = computeBilinerWeights(u)
                AtWbnbp = [0, 0]
                for y in range(self.winsize[1]):
                    for x in range(self.winsize[0]):
                        gx = iu[0] + x
                        gy = iu[1] + y

                        ## TODO 3.2
                        ## Compute the following parts of step 2
                        ## AtWbnbp    2 x 1 vector

                        bnext_bprev = nextImg[gy, gx] - prevImg[gy, gx]

                        # Compute AtWbnbp = A.T * W * (bnext - bprev)
                        AtWbnbp[0] += bw[y, x] * prevDerivx[gy, gx] * bnext_bprev
                        AtWbnbp[1] += bw[y, x] * prevDerivy[gy, gx] * bnext_bprev

                AtWA = np.matmul(A.transpose(), A)

                ## TODO 3.2
                ## - Solve the linear system for deltaU: At * W * A * deltaU = - At * W * (bnext - bprev)
                ## - Add deltaU to u
                ## - Implement the early termination condition (Step 4)

                deltaU = -np.dot(invAtWA, AtWbnbp)

                # Update u with deltaU
                u += deltaU.flatten()

                # Termination condition (early stop if displacement is small)
                if np.linalg.norm(deltaU) < 1e-4:
                    break

            nextPts[ptidx] = u + halfWin

        return nextPts, status
