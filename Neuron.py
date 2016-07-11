
import pandas as pd
from numpy import *
from Logging import Log

from numpy import eye, zeros, ones, dot, exp
from numpy.random import rand, randn
from numpy.linalg import inv


###############################################################
#                                                             #
#                            LNU                              #
#                                                             #
###############################################################

class LNUGD:

  w = []

  def countSerie(self, Y, X, **kwargs):

    #volitelne parametry
    prediction =  kwargs.get('prediction', 1)
    mu = kwargs.get('mu', 0.1)
    self.w = kwargs.get('weigths', self.w)
    logging = kwargs.get('logging', False)

    log = Log(logging, "LNU.count()")
    log.message("start")

    
    #pro vypocet
    nx = len(X) * 4 + 1 #lenX * X width + 1
    nw = nx
    N=len(Y)
    e=zeros(N) # "prazdna " array pro vypocet chyb v delce dat

    yn = Y.copy() # init vystup neuronu (site)
    yn[0] = float('nan')
    log.message(yn[1])


    x = [1.]
    x = array(x)
    if len(self.w) < 1:
      self.w = random.randn(nw)/nw
      print self.w

      self.w = random.randn(nw)
      print self.w

    #pro zaznam
    Wall = []
    Wall.append(self.w) #inicializovane vahy
    MSE = []


    for k in range(N-prediction):
      ##vstupni vektor
      x = [1.]
      for i in range(len(X)):
        x = concatenate((x, [ X[i][k], X[i][k-1], X[i][k-2], X[i][k-3] ]))

      #vypocet neuronu
      yn[k+prediction]=dot(self.w,x)


      if not isnan(yn[k]):
        e[k]=Y[k]-yn[k]

        #update vah
        dydw=x     # pro LNU dy/dw = x
        dw=mu/(1+sum(x**2))*e[k]*dydw  # normalizovany GD
        self.w=self.w+dw

      #pro vykreselni vah 
      Wall.append(self.w)
    
    MSE.append(sum(e**2)/N)    
    
    log.message("done \n")
    return yn, self.w, Wall, MSE, e


  def train(self, Y_train, X_train, **kwargs):

    prediction =  kwargs.get('prediction', 1)
    epochs = kwargs.get('epochs', 1)
    logging = kwargs.get('logging', False)

    log = Log(logging, "LNU.train()")
    log.message("start")

    Wall = []
    MSE = []
    w = []

    #Training
    for epoch in range(epochs):

      yn, w, Wall0, MSE0, e = self.countSerie(Y_train, X_train, prediction = prediction)

      Wall.append(Wall0)
      MSE.append(MSE0)

    log.message("done \n")
    return yn, w, e, Wall, MSE


###############################################################
#                                                             #
#                            QNU                              #
#                                                             #
###############################################################


class QNULM:

  w = []

  def countSerie(self, Y, X, **kwargs):

    #volitelne parametry
    prediction =  kwargs.get('prediction', 1)
    mu = kwargs.get('mu', 0.9)
    self.w = kwargs.get('weigths', self.w)
    logging = kwargs.get('logging', False)

    log = Log(logging, "MLP.phiCell()")
    log.message("start")


    N=len(Y)   # X.shape[0]
    e=zeros(N)
    yn=Y.copy()  # init vystup neuronu (site)
    nx = len(X) * 3 + 1 #lenX * X width + 1
    nw=(nx*nx+nx)/2
    if len(self.w) < 1:
      self.w = random.randn(nw)/nw
    x=[]
    colx=[]
    J=zeros((N,nw))
    I=eye(nw)

    #pro zaznam
    MSE = [] # prumerna kv. chyb


    for k in range(N-prediction):

      #vstupni vektor
      x = [1.]
      for i in range(len(X)):
        x = concatenate((x, [ X[i][k], X[i][k-1] , X[i][k-2]]))#, X[i][k-3], X[i][k-4], X[i][k-5] ]))
      
      colx = []
      for i in range(nx):
        for j in range(i,nx):
          colx.append(x[i]*x[j])

      yn[k+prediction]=dot(self.w,colx)

      e[k]=Y[k]-yn[k]
      dydw=colx       # pro QNU a vyssi HONU (Higher Order Neural Units)
      J[k,:]=dydw

    print "___"
    print len(dot(linalg.inv(dot(J.T,J)+1.0/mu*I),J.T))
    print len(e)
    print len(inv(dot(J.T,J)+1./mu*I))
    dw=dot(dot(linalg.inv(dot(J.T,J)+1.0/mu*I),J.T),e)
    self.w=self.w+dw
    MSE.append(sum(e**2)/N)  
    
    return yn, self.w, MSE, e


  def train(self, Y_train, X_train, **kwargs):

    prediction =  kwargs.get('prediction', 1)
    epochs = kwargs.get('epochs', 1)
    logging = kwargs.get('logging', False)

    log = Log(logging, "QNU.train()")
    log.message("start")

    Wall = []
    MSE = []

    #Training
    for epoch in range(epochs):

      yn, w, MSE0, e = self.countSerie(Y_train, X_train, prediction = prediction)

      Wall.append(w)
      MSE.append(MSE0)

    log.message("done \n")
    return yn, w, e, Wall, MSE


###############################################################
#                                                             #
#                            RBF                              #
#                                                             #
###############################################################

class RBF:

  def fphi(self, nu,beta):
    phi=exp(-nu/beta)  #nu^2 pokud by mmohlo byt nu zaporne
    return(phi)

  def fnu(self, w,x):
    return(sqrt(sum((w-x)**2)))

  def train(self, Y_train, X_train, **kwargs):

    prediction =  kwargs.get('prediction', 1)
    logging = kwargs.get('logging', False)
    log = Log(logging, "RBF.train()")
    log.message("start")

    self.W = []
    self.Y = []

    # co radek to neuron, centra RBF funkci RBF neurnu
    for k in range(len(X_train[0])):
      self.W.append([])
      for i in range(len(X_train)):
        l = k - prediction
        self.W[k].append([ X_train[i][l] ])#, X_train[i][l-1], X_train[i][l-2], X_train[i][l-3], X_train[i][l-4] ])
      self.Y.append(Y_train[k])

    log.message("done \n")

  def count(self, Y, X, **kwargs):

    prediction =  kwargs.get('prediction', 1)
    beta =  kwargs.get('beta', 0.1)
    logging = kwargs.get('logging', False)

    log = Log(logging, "RBF.count()")
    log.message("start")

    N=len(Y)
    self.Yn=Y.copy()
    e=zeros(N)
    colx = []
    allColx = []

    for j in range(N-prediction):

      log.message(j, conditioned=True)

      #update neuronu LIFO (moving window) start updating after prediction lag
      #if (j > 0 and j > prediction):
      #  self.W = self.W[1:] #remove first
      #  self.W.append(allColx[-prediction]) 
      #  self.Y = self.Y[1:] #remove first
      #  self.Y.append(Y[j])

      colx = []
      for i in range(len(X)):
        colx.append([X[i][j] ])#, X[i][j-1], X[i][j-2], X[i][j-3], X[i][j-4]])

      allColx.append(colx)

      Nw=len(self.W) #pocet radku je poc neuronu
      phi=zeros(Nw)
      nu=zeros(Nw)
      for i in range(Nw):
          nu[i]=self.fnu(asarray(self.W[i]),asarray(colx)) #nu[i]=fnu(W[i,:],x)
      phi=self.fphi(nu,beta) # vystupy RBF = neuronu

      self.Yn[j+prediction]=sum(asarray(phi)*asarray(self.Y))/sum(asarray(phi))

    log.message("done \n")
    return self.Yn


###############################################################
#                                                             #
#                            MLPML                            #
#                                                             #
###############################################################

class MLPLM:

  w = []
  v = []

  def phi(self, ny):
    bz = 2. / ( 1. + exp( -ny ) ) -1.
    return bz

  def dphidny(self, ny):
    bzz = 2. * exp( -ny ) / ( 1. + exp( -ny ) ) **2
    return bzz

  def count(self, Y, X, **kwargs):


    prediction =  kwargs.get('prediction', 1)
    epochs = kwargs.get('epochs', 1)
    logging = kwargs.get('logging', False)

    log = Log(logging, "MLPGD.count()")
    log.message("start")
   
    yr = Y

    muw = 0.05
    muv = 0.01

    N = len(Y)
    n1 = 2
    nx = len(X) * 4 + 1
    nv = 1 + n1
    nxi = nv

    if len(self.w) < 1:
      self.w = randn( n1, nx ) / nx

    if len(self.v) < 1:
      self.v = randn( nv ) / nv
      
    e = zeros( N )
    y = Y.copy()
    #x = ones( nx )
    xi = ones( nxi )
    dxidny = zeros ( n1 + 1 )
    dydv = zeros(( N, nv ))
    dydw = zeros( ( N, nx, n1 ) )
    Lv = eye( nv )
    Lw = eye( nx )
    SSE = zeros( epochs )

    for epoch in range( epochs ):
      for k in range ( N - prediction):

        #vstupni vektor
        x = [1.]
        for i in range(len(X)):
          x = concatenate((x, [ X[i][k], X[i][k-1], X[i][k-2], X[i][k-3] ]))

        ny = dot( self.w, x )


        xi[1:] = self.phi(ny)
        y[ k + prediction ] = dot( self.v, xi )
        e[ k ] = yr[ k ] - y[ k - prediction]


        #vahy vystupniho neuronu
        dydv[k, :] = xi

        #vahy skrytych neuronu
        dxidny[1:] = self.dphidny(ny)
        for i in range(1, n1 + 1 ):
          dydw[k, :, i-1 ] = self.v[ i ] * dxidny[ i ] * x

      Jv = dydv
      dv = dot( dot( inv( dot( Jv.T, Jv ) + 1. / muv * Lv ), Jv.T ), e )
      self.v = self.v + dv

      for i in range( 1, n1 + 1 ):
        Jw = dydw[ :, :, i - 1 ]
        dw = dot( dot( inv( dot( Jw.T, Jw ) + 1. / muw * Lw ), Jw.T ), e )
        self.w[ i - 1, : ] = self.w[ i - 1, : ] + dw

      SSE[ epoch ] = dot( e, e )
        


    return y


###############################################################
#                                                             #
#                            MLPMLWL                          #
#                                                             #
###############################################################

class MLPLMWL:

  w = []
  v = []

  def phi(self, ny):
    bz = 2. / ( 1. + exp( -ny ) ) -1.
    return bz

  def dphidny(self, ny):
    bzz = 2. * exp( -ny ) / ( 1. + exp( -ny ) ) **2
    return bzz

  def count(self, Y, X, **kwargs):


    prediction =  kwargs.get('prediction', 1)
    epochs = kwargs.get('epochs', 1)
    logging = kwargs.get('logging', False)
    lw = kwargs.get('learningWindow', 0)
    ol = kwargs.get('overLearn', 0)

    log = Log(logging, "MLPGD.count()")
    log.message("start")
   
    yr = Y

    muw = 0.05
    muv = 0.01

    N = len(Y)
    nx = len(X) * 4 + 1
    n1 = 5
    nv = 1 + n1
    nxi = nv

    if len(self.w) < 1:
      self.w = randn( n1, nx ) / nx

    if len(self.v) < 1:
      self.v = randn( nv ) / nv
      
    e = zeros( lw )
    y = Y.copy()
    #x = ones( nx )
    xi = ones( nxi )
    dxidny = zeros ( n1 + 1 )
    dydv = zeros(( lw, nv ))
    dydw = zeros(( lw, nx, n1 ))
    Lv = eye( nv )
    Lw = eye( nx )
    SSE = zeros( epochs )

    for epoch in range( epochs ):
      for k in range ( N - prediction):

        #vstupni vektor
        x = [1.]
        for i in range(len(X)):
          x = concatenate((x, [ X[i][k], X[i][k-1], X[i][k-2], X[i][k-3] ]))

        ny = dot( self.w, x )


        xi[1:] = self.phi(ny)
        y[ k + prediction ] = dot( self.v, xi )
        ek = yr[ k ] - y[ k ]

        e = concatenate(( e, [ek] ))
        e = delete(e, 0, 0)


        #vahy vystupniho neuronu
        dydv = delete(dydv, 0, 0)
        dydv = vstack(( dydv, xi ))

        #vahy skrytych neuronu
        dxidny[1:] = self.dphidny(ny)
        dydwk = zeros( ( 1, nx, n1 ) )
        for i in range(1, n1 + 1 ):
          dydwk[0, :,i-1] = self.v[ i ] * dxidny[ i ] * x
        dydw = delete(dydw, 0, 0)
        dydw = vstack(( dydw, dydwk))

        if k > lw and k % ol == 0:
          print k
          Jv = dydv
          dv = dot( dot( inv( dot( Jv.T, Jv ) + 1. / muv * Lv ), Jv.T ), e )
          self.v = self.v + dv

          for i in range( 1, n1 + 1 ):
            Jw = dydw[ :, :, i - 1 ]
            dw = dot( dot( inv( dot( Jw.T, Jw ) + 1. / muw * Lw ), Jw.T ), e )
            self.w[ i - 1, : ] = self.w[ i - 1, : ] + dw

      SSE[ epoch ] = dot( e, e )
        


    return y

###############################################################
#                                                             #
#                            MLPGD                            #
#                                                             #
###############################################################

class MLPGD:

  w = []
  v = []

  def phi(self, ny):
    bz = 2. / ( 1. + exp( -ny ) ) -1.
    return bz

  def dphidny(self, ny):
    bzz = 2. * exp( -ny ) / ( 1. + exp( -ny ) ) **2
    return bzz

  def count(self, Y, X, **kwargs):


    prediction =  kwargs.get('prediction', 1)
    epochs = kwargs.get('epochs', 1)
    logging = kwargs.get('logging', False)

    log = Log(logging, "MLPGD.count()")
    log.message("start")
   
    yr = Y

    muw = 0.05
    muv = 0.01

    N = len(Y)
    n1 = 100
    nx = len(X) * 4 + 1
    nv = 1 + n1
    nxi = nv

    if len(self.w) < 1:
      self.w = randn( n1, nx ) / nx

    if len(self.v) < 1:
      self.v = randn( nv ) / nv
      
    e = zeros( N )
    y = Y.copy()
    #x = ones( nx )
    xi = ones( nxi )
    dxidny = zeros ( n1 + 1 )
    dydv = zeros(( N, nv ))
    dydw = zeros( ( N, nx, n1 ) )
    SSE = zeros( epochs )

    for epoch in range( epochs ):
      for k in range ( N - prediction):

        #vstupni vektor
        x = [1.]
        for i in range(len(X)):
          x = concatenate((x, [ X[i][k], X[i][k-1], X[i][k-2], X[i][k-3] ]))

        ny = dot( self.w, x )

        xi[1:] = self.phi(ny)
        y[ k + prediction ] = dot( self.v, xi )
        e[ k ] = yr[ k ] - y[ k ]

        #vahy vystupniho neuronu
        dydv[k, :] = xi
        dv = muv/(1+sum(xi**2))*e[k]*dydv[k]  # normalizovany GD
        self.v = self.v + dv

        #vahy skrytych neuronu
        dxidny[1:] = self.dphidny(ny)
        for i in range(1, n1 + 1 ):
          dydw[k, :, i-1 ] = self.v[ i ] * dxidny[ i ] * x
        
        for i in range( 1, n1 + 1 ):
          dw = muw/(1+sum(x**2))*e[k]*dydw[ k, :, i - 1 ]
          self.w[ i - 1, : ] = self.w[ i - 1, : ] + dw

    return y


class MLPELM:

  w = []
  v = []

  def phi(self, ny):
    bz = 2. / ( 1. + exp( -ny ) ) -1.
    return bz

  def dphidny(self, ny):
    bzz = 2. * exp( -ny ) / ( 1. + exp( -ny ) ) **2
    return bzz

  def count(self, Y, X, **kwargs):


    prediction =  kwargs.get('prediction', 1)
    epochs = kwargs.get('epochs', 1)
    logging = kwargs.get('logging', False)
    learning = kwargs.get('learning', True)

    log = Log(logging, "MLPGD.count()")
    log.message("start")
   
    yr = Y

    #muw = 0.05
    #muv = 0.01
    muv = 0.005

    N = len(Y)
    n1 = 400
    nx = len(X) * 4 + 1
    nv = 1 + n1
    nxi = nv

    if len(self.w) < 1:
      self.w = randn( n1, nx ) / nx

    if len(self.v) < 1:
      self.v = randn( nv ) / nv
      
    e = zeros( N )
    y = Y.copy()
    #x = ones( nx )
    xi = ones( nxi )
    dxidny = zeros ( n1 + 1 )
    dydv = zeros(( N, nv ))
    dydw = zeros( ( N, nx, n1 ) )
    SSE = zeros( epochs )

    for epoch in range( epochs ):
      for k in range ( N - prediction):

        #vstupni vektor
        x = [1.]
        for i in range(len(X)):
          x = concatenate((x, [ X[i][k], X[i][k-1], X[i][k-2], X[i][k-3] ]))

        ny = dot( self.w, x )

        xi[1:] = self.phi(ny)
        y[ k + prediction ] = dot( self.v, xi )
        e[ k ] = yr[ k ] - y[ k ]

        #vahy vystupniho neuronu
        dydv[k, :] = xi
        dv = muv/(1+sum(xi**2))*e[k]*dydv[k]  # normalizovany GD
        if learning:
          self.v = self.v + dv

        #vahy skrytych neuronu
        #dxidny[1:] = self.dphidny(ny)
        #for i in range(1, n1 + 1 ):
        #  dydw[k, :, i-1 ] = self.v[ i ] * dxidny[ i ] * x
        
        #for i in range( 1, n1 + 1 ):
        #  dw = muw/(1+sum(x**2))*e[k]*dydw[ k, :, i - 1 ]
        #  self.w[ i - 1, : ] = self.w[ i - 1, : ] + dw

    return y


