# DBSCAN.py
# DBSCAN that uses different depth-based functions as the distance measure

import numpy as np
import sys

class euclideanDistance:
    def depth(self, p1, p2):
        ''' 
        * p1, p2: Points represented as numpy arrays
        
        **returns**: The euclidean distance between two points
        '''
        point = np.subtract(p1, p2)
        distance = np.linalg.norm(point) 

        # Depth is opposite of distance, where two points close together have a high depth and a low distance. 
        # So the distance is made negative. Use the theta parameter like epsilon for dbscan but just give a negative value instead
        return distance * -1 

class mahalanobisDepth:
    def __init__(self):
        self.inverseCovar = None

    def depth(self, p1, p2):
        ''' 
        * p1, p2: Points represented as numpy arrays
        * inverseCovar: The inverse of the covariance matix for the distribution
        
        **returns**: The mahalanobis depth distance between two points in the distribution
        '''

        point = np.subtract(p1, p2)
        tmp = np.matmul(point, self.inverseCovar)
        tmp = np.matmul(tmp, point.T) 
        tmp = 1 / (1 + tmp)

        return tmp

class projectionDepth:
    def __init__(self):
        self.maxUnitVec = None
        self.MADalongMUV = None

    def depth(self, p1, p2):
        ''' 
        * p1, p2: Points represented as numpy arrays
        * maxUnitVec: The unit vector in the direction of max variance for the distribution
        * MADalongMUV: The Mean Absolute Deviation along the Max Unit Vector for the distribution
        
        **returns**: The projection depth distance between two points in the distribution
        '''

        # points are vectors
        # p1Proj = np.dot( p1, self.maxUnitVec )
        # p2Proj = np.dot( p2, self.maxUnitVec )
        # diff = p1Proj - p2Proj
        # if diff < 0: diff = diff * -1
        
        vec = np.subtract(p2,p1)
        eucDist = np.linalg.norm(vec)
        
        return 1 / (1 + (eucDist / self.MADalongMUV))

    def compute1DMedianAbsoluteDeviation(self, data, unitVector):
        # Get the 1D projection of the points along the unit vector
        proj = np.dot(data, unitVector)
        #print('point projection\n', proj)

        med = np.median(proj, axis = None)
        #print('median', med)

        proj = proj - med
        proj = np.absolute(proj)
        #print('proj-med', proj)

        mad = np.median(proj, axis = None, overwrite_input = True)

        self.maxUnitVec = unitVector
        self.MADalongMUV = mad

        return mad

def computeCovarianceMatric(inmat):
    #print(inmat)
    rows, cols = inmat.shape

    covar = np.cov(inmat.T)
    #print( 'covariance matrix:\n', covar )
    
    w,v = np.linalg.eigh( covar )
    #print('eigs:\n',w,'\n', v)

    inversecovar = np.linalg.pinv( covar )
    #print( 'inverse covariance matrix:\n',inversecovar)

    return covar, w, v

def dbscan( data,  CL, theta, distObj, minPts = 3, startLabel = 1 ):
    '''
    **data**: A Python iterable object containing the points
    **CL**: A parallel array with the input data points. Each entry is the cluster the point is assigned to.
    **theta**: The cluster depth threshold
    **distObj**: A distance object used to calculate the distance/depth with the approriate function
    **minPts**: The minimum number of points needed for a core point
    **startLabel**: The label to start the reclustering with
    '''

    data = [tuple(p) for p in data] # Convert to tupels for set usage
    #data = [(p[0], p[1]) for p in data] # Convert to tupels for set usage
    
    label = startLabel
    for i, point in enumerate(data):
        if CL[i] == None or CL[i] == 0:  # We have an unclustered point
            # Get neighbors of the point
            currentCore = [(i, point)]

            for j, p2 in enumerate(data):
                if CL[j] == None or CL[j] == 0:
                    if i == j: continue

                    val = distObj.depth(point, p2)
                    if val > theta:
                        currentCore.append( (j, p2) )

            # check if we have a core group
            # If not, they are all noise for now  
            if len(currentCore) < minPts:
                for index, p2 in currentCore: 
                    CL[index] = 0

                continue

            # If we get here, we have a core group. So grow it.
            CL[i] = label

            coreSet = set(currentCore)
            while len(coreSet) > 0:
                index, coreP = coreSet.pop()

                if CL[index] == 0: # Previosly made noise
                    CL[index] = label # Add border point

                if CL[index] != None: continue

                CL[index] = label
                # Find all neighbors of P
                newCore = [(index, coreP)]
                for j, p2 in enumerate(data):
                    if CL[j] == None or CL[j] == 0 or CL[j] == label:
                        if index == j: continue

                        val = distObj.depth(coreP, p2)

                        if val > theta:
                            newCore.append( (j, p2) )

                # Update the core
                if len(newCore) >= minPts:
                    coreSet |= set(newCore)

            label += 1

    return label

def dbscanLaunch(inputData, theta, distObj, minPts = 3, startLabel = 1):
    '''
    **inputData**: A Python iterable object containing the points
    **theta**: The cluster depth threshold
    **distObj**: A distance object used to calculate the distance/depth with the approriate function
    **minPts**: The minimum number of points needed for a core point
    **startLabel**: The label to start the reclustering with
    '''

    rows, cols = inputData.shape
    CL = [None] * rows

    lastLabel = dbscan(inputData, CL, theta, distObj, minPts, startLabel)

    return CL, lastLabel

def localClusterReScan( inmat, G_eigs, G_eigVecs, theta, covarSimVal,  CL, distObj, minPts, nextLabel ):
    '''
    **inmat**: the input points (numpy array Nx2 array)
    **G_eigs**: the eigenvalues of the global covariance matrix
    **G_eigVecs**: the eigenvectors of the global covariance matrix
    **theta**: threshold.  Should be the same as used for a global clustering
    **covarSimVal**: Covariance Similarity Value. 1 reclusters everything.  0 reclusters nothing.
    **CL**: the cluster labels for the global clustering
    **distObj**: A distance object used to calculate the distance/depth with the approriate function
    **minPts**: The minimum number of points needed for a core point
    **nextLabel**:  The label to start the reclustering with.
    '''

    result = []
    newPoiOrder = []
    # Get the unique labels 
    uniqueLabels = list(set(CL))
    for x in uniqueLabels:
        #print('!!!!! Processing LABEL CLUSTER: ', x)

        # Get the list of points on that label
        pois = [row for i, row in enumerate(inmat) if CL[i] == x]

        if len(pois) <= minPts :
            result += [x] * len(pois)
            newPoiOrder.extend( pois )
            continue

        if x == 0:
            result += [x] * len(pois)
            newPoiOrder.extend( pois )
            continue

        newMat = np.array(pois)

        # Now set up a scaled recluster on that cluster
        covar, localEigs, localEigVecs =  computeCovarianceMatric( newMat )

        #compute covar comparison statistics
        # mult d1x2, d2x1.  need eigs of original covars
        d1x2 = np.matmul( inmat, localEigVecs )
        d2x1 = np.matmul( newMat, G_eigVecs )
        d1x2covar = np.cov( d1x2.T)
        d2x1covar = np.cov( d2x1.T)
        # get the diagonals of the covar in an array
        # compute eigs as percent of total variance
        d1x2eigs = np.diag( d1x2covar )
        d2x1eigs = np.diag( d2x1covar )
        localNormEigs = localEigs / (np.sum( localEigs ) )
        globalNormEigs = G_eigs / (np.sum( G_eigs ) )
        d1x2NormEigs = d1x2eigs/ np.sum(d1x2eigs )
        d2x1NormEigs = d2x1eigs/ np.sum(d2x1eigs )
        #print( 'variance explained')
        #print('local ', localNormEigs)
        #print('global', globalNormEigs )
        #print('d1x2  ', d1x2NormEigs)
        #print('d2x1  ', d2x1NormEigs)
        #print('++++')
        # compute the S1 statistic
        simVal = 0
        for i,e in enumerate( localNormEigs ):
            x1 = globalNormEigs[i] - d2x1NormEigs[i]
            x1 *= x1
            x2 = d1x2NormEigs[i] - localNormEigs[i]
            x2 *= x2
            simVal += x1+x2
        simVal *=2
        simVal /= 8
        simVal = 1-simVal
        print('++++++', covarSimVal, simVal )
        if  covarSimVal <= simVal:
            result += [x] * len(pois)
            newPoiOrder.extend( pois )
            continue
 
        # Now lets scale the new covariance matrix to equalize the area of the data elipses
        lprod = np.prod(localEigs)
        if lprod == 0: lprod = 0.0000001
        scale = np.prod(G_eigs)/lprod
        # (G_eigs[0] * G_eigs[1]) / (localEigs[0] * localEigs[1])
        
        # need to generalize dimension
        dims = len( localEigs )
        sq = np.power([scale],[1.0/dims] )[0]
        scaleMat = np.ones( (dims, dims) )
        np.fill_diagonal( scaleMat, sq)
        #print ('scalemat:', scaleMat)
        #print( 'orig covar')
        #print( covar )
        covar = np.multiply(covar, scaleMat)
        #print('scaled covar:')
        #print( covar )
        

        distObj.inverseCovar = np.linalg.pinv( covar )

        # Now we can DB that scan again, with the scaled covar
        newCL, nextLabel  = dbscanLaunch(newMat, theta, distObj, minPts, nextLabel)

        # Concatenate the two arrays together eventually giving the new CL result
        result += newCL
        newPoiOrder.extend( pois )
        
        # print('>>>>>>> new clustering:\n')
        # for i, row in enumerate( newMat):
        #     print( i, row, newCL[i])

    return result, newPoiOrder
