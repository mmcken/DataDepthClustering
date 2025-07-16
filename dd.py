from collections import deque
import numpy as np
import sys

if len(sys.argv) < 2:
    print( 'usage: exe inputFile')
    exit()
points = []
path = "Datasets/"
with open(path + sys.argv[1], 'r') as infile:
    infile.readline()

    for line in infile:
        if len(line) < 2:
            continue
        nums = line.split(",")
        points.append( (float( nums[0] ), float( nums[1])) )


def computeCovarianceMatric( inmat ):

    print( inmat )
    rows, cols = inmat.shape

    ######## manual covariance computation
    ### just use numpy below this
    ##compute the centroid.  sum up the colums and divide
    #sums = np.sum( inmat, axis=0 )
    #centroid = [sums[0]/rows, sums[1]/rows]
    #centroid = np.array( centroid)
    #print('centroid:', centroid)

    ## compute the covariance matrix
    #cenMat = [ centroid for x in range(rows)   ]
    #cenMat = np.array(cenMat)
    #diff = inmat - cenMat
    #covar = np.matmul( diff.T, diff )
    #print (covar)

    ## average in inverse the covariance matrix
    #covar = np.divide( covar, rows)
    ##########################3
    covar = np.cov(inmat.T)
    print( 'covariance matrix:\n', covar )
    
    w,v = np.linalg.eigh( covar )
    print('eigs:\n',w,'\n', v)

    inversecovar = np.linalg.inv( covar )
    print( 'inverse covariance matrix:\n',inversecovar)

    ###
    ##compute some pairwise depth vals.
    #print('pairwise depths')
    #p1 = np.array([3,13])
    #p2 = np.array([4,13])
    #p = np.subtract( p1, p2 )
    #print( p1,p2, p, MD( p, inversecovar) )
    #p1 = np.array([7,13])
    #p2 = np.array([8,13])
    #p = np.subtract( p1, p2 )
    #print( p1,p2, p, MD( p, inversecovar) )
    #p1 = np.array([3,13])
    #p2 = np.array([3.5,13])
    #p = np.subtract( p1, p2 )
    #print( p1,p2, p, MD( p, inversecovar) )

    #p1 = np.array([3,15])
    #p2 = np.array([3,16])
    #p = np.subtract( p1, p2 )
    #print( p1,p2, p, MD( p, inversecovar) )
    #p1 = np.array([3,19])
    #p2 = np.array([3,20])
    #p = np.subtract( p1, p2 )
    #print( p1,p2, p, MD( p, inversecovar) )
    ####
    return covar, w, v

class mahanalobisDepth:
    def __init__( self ):
        self.inverseCovar = None

    def depth( self, p1,p2  ):
        point = np.subtract(p1,p2)
        tmp = np.matmul( point, self.inverseCovar)
        tmp = np.matmul(tmp, point.T) 
        tmp = 1/(1+tmp)
        return tmp




#ok,  it plays out.  

def dbscan2( data, dataDict, CL, theta, distObj, minPts = 3, startLabel = 1 ):
    '''
    **data**:  a Python iterable object containg the points
    **dataDict**: a dictionary mapping tuples of points to their original index in the array
    **CL**:  a parallel array with the input data points.  Each entry is the cluster the point is assigned to.
    **covar**: the covariance matrix for the data
    **theta**: the cluster depth threshold.
    **minPts**: the minimum number of points needed for a core point
    '''
    data = [(p[0],p[1]) for p in data] # convert ot tupels for set usage
    label = startLabel
    for i,point in enumerate( data ):
        if CL[i] == None or CL[i] == 0:  #we have an unclustered point
            #get neighbors of the point
            currentCore = [(i,point)]
            for j,p2 in enumerate(data):
                if i == j: continue
                val = distObj.depth( point, p2 )
                if val > theta:
                    currentCore.append( (j,p2) )
            # check if we have a core group
            # if not, they are all noise for now  
            if len( currentCore ) < minPts:
                for index,p2 in currentCore: 
                    CL[index] = 0
                continue
            # if we get here,  we have a core group. so grow it
            CL[i] = label
            coreSet = set(currentCore)
            while len( coreSet ) > 0:
                index,coreP = coreSet.pop()
                if CL[ index ] == 0: # previosly made noise
                    CL[ index ] = label #add border point
                if CL[ index ] != None: continue
                CL [index] = label
                # find all neighbors of P
                newCore = []
                for j, p2 in enumerate(data):
                    if index == j: continue
                    val = distObj.depth( coreP, p2 )
                    if val > theta:
                        newCore.append( (j,p2) )
                # update the core
                if len(newCore ) >= minPts:
                    coreSet |= set( newCore )
            label+=1
    return label


def dbscanLaunch( inputData, theta, distObj, minPts = 3, startLabel = 1 ):
    rows,cols = inputData.shape
    CL = [None] * rows
    dataCopy = [(r[0],r[1]) for r in inputData ]
    dataDict = dict()
    for i,row in enumerate( dataCopy):
        dataDict[row] = i
    que = deque( dataCopy )

    lastLabel = dbscan2( inputData, dataDict, CL, theta, distObj, minPts, startLabel )
    return CL, lastLabel

def localClusterReScan( inmat, G_eigs, G_eigVecs, theta, CL, distObj, minPts, nextLabel ):
    '''
    **inmat**: the input points (numpy array Nx2 array)
    **G_covar**: the global covariance matrix
    **G_eigs**: the eigenvalues of the global covariance matrix
    **G_eigVecs**: the eigenvectors of the global covariance matrix
    **theat**: threshold.  Should be the same as used for a global clustering
    **CL**: the cluster labels for the global clustering
    **nextLabel**:  The label to start the reclustering with.
    '''
    # get the unique labels 
    uniqueLabels = list(set(CL))
    for x in uniqueLabels:
        print('!!!!! Processing LABEL CLUSTER: ', x)
            #get the list of points on that label
        pois = [(row[0], row[1]) for i,row in enumerate( inmat ) if CL[i] ==x]
        newMat = np.array( pois )
        # now set up a scaled recluster on that cluster
        rows,cols = newMat.shape
        covar, localEigs, localEigVecs =  computeCovarianceMatric( newMat )
        # now lets scale the new covariance matrix to equalize the area 
        # of the data elipses
        scale = (G_eigs[0]*G_eigs[1])/(localEigs[0]*localEigs[1])
        print('globabl eigs:\n', globalEigs)
        print('localEigs:\n', localEigs)
        print('scale: ', scale)
        sq = np.sqrt([scale])[0]
        print('sqrt scale:', sq)
        scaleMat = np.array([[sq,1],[1,sq]])
        print('scaleMat:\n', scaleMat)
        print('unscaled covar:\n ', covar)    
        covar = np.multiply( covar, scaleMat)
        print('scaled covar:\n', covar)
        distObj.inverseCovar = np.linalg.inv( covar )

        # now we can DB that scan again,  with the scaled covar
        newCL, nextLabel  = dbscanLaunch( newMat, theta, distObj, minPts, nextLabel)
        print('>>>>>>> new clustering:\n')
        for i, row in enumerate( newMat):
            print( i, row, newCL[i])


#create covariance matrix
inmat = np.array( points)
covar, globalEigs, globalEigVecs =  computeCovarianceMatric( inmat )

# now lets db that scan
theta = 0.7
minPts = 3
MD = mahanalobisDepth()
MD.inverseCovar = np.linalg.inv( covar )

CL, lastLabel  = dbscanLaunch( inmat, theta,MD, minPts)
for i, row in enumerate( inmat):
    print( i, row, CL[i])

localClusterReScan( inmat, globalEigs, globalEigVecs, theta, CL, MD, minPts, lastLabel)

class projectionDepth:
    def __init__( self):
        self.maxUnitVec = None
        self.MADalongMUV = None

    def depth( self, p1, p2 ):
        ''' 
        * p1, p2:   points represented as numpy arrays
        * maxUnitVec:   the unit vector in the direction of max variance for the distribution. 
        * MADalongMUV:  the Mean Absolute Deviation along the Max Unit Vector for the distribution
        
        **returns:** the projection depth distance between two points in the distribution
        '''
        # points are vectors
        # p1Proj = np.dot( p1, self.maxUnitVec )
        # p2Proj = np.dot( p2, self.maxUnitVec )
        # diff = p1Proj - p2Proj
        # if diff < 0: diff = diff * -1
        

        vec = np.subtract(p2,p1)
        eucDist = np.linalg.norm(vec)
        
        return 1/(1+(eucDist / self.MADalongMUV))

    def compute1DMedianAbsoluteDeviation(self, data, unitVector ):
        # get the 1D projection of the points along the unit vector
        proj = np.dot( data, unitVector )
        print('point projection\n', proj)
        med = np.median( proj, axis = None )
        print('median', med)
        proj = proj - med
        proj = np.absolute( proj)
        print('proj-med', proj)
        mad = np.median( proj, axis = None, overwrite_input = True)
        self.maxUnitVec = unitVector
        self.MADalongMUV = mad
        return mad

# for proj depth, we get the eigenvector with max eigenval
maxEigVec = globalEigVecs[:,-1]
#maxEigVec =np.array([maxEigVec])
#maxEigVec = maxEigVec.T
print( 'max eig, global eigs\n',maxEigVec)
print(globalEigVecs )
PD = projectionDepth()
PD.compute1DMedianAbsoluteDeviation( inmat, maxEigVec)
for p in inmat:
    print( p, "(7,17): ",PD.depth( p, (7,17)))
theta = .64
CL, lastLabel = dbscanLaunch( inmat, theta, PD, minPts)
print( "PROJ Depth Scan")
for i, row in enumerate( inmat):
    print( i, row, CL[i])


