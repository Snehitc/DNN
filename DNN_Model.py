import numpy as np


# Activation
def activation(function, Z):
    
    if function == "sigmoid":
        A = 1/(1+np.exp(-Z))
        
    elif function == "tanh":
        A = (1-np.exp(-2*Z))/(1+np.exp(-2*Z))
        
    elif function == "relu":
        A =  Z * (Z > 0)

    return A


# derivative
def diff(function, Z):
    if function == "sigmoid":
        gZ = (1/(1+np.exp(-Z))) * (1-(1/(1+np.exp(-Z))))
        
    elif function == "tanh":
        gZ = 1 - (((1-np.exp(-2*Z))/(1+np.exp(-2*Z)))**2)

    elif function == "relu":
        gZ =  1 * (Z > 0)

    return gZ



# 1st Random Forward propogate
#layer 1
W1 = np.random.randn(nL1,nx)*np.sqrt(2./nx)
b1 = np.zeros((nL1,1))

#Layer 2
W2 = np.random.randn(nL2,nL1)*np.sqrt(2./nL1)
b2 = np.zeros((nL2,1))

#layyer 3
W3 = np.random.randn(nL3,nL2)*np.sqrt(2./nL2)
b3 = np.zeros((nL3,1))

#layyer 4
W4 = np.random.randn(nL4,nL3)*np.sqrt(2./nL3)
b4 = np.zeros((nL4,1))



# Initial momentum and RMSprop
VdW1 = SdW1 = np.zeros((nL1,nx))
Vdb1 = Sdb1 = np.zeros((nL1,1))

VdW2 = SdW2 = np.zeros((nL2,nL1))
Vdb2 = Sdb2 = np.zeros((nL2,1))

VdW3 = SdW3 = np.zeros((nL3,nL2))
Vdb3 = Sdb3 = np.zeros((nL3,1))

VdW4 = SdW4 = np.zeros((nL4,nL3))
Vdb4 = Sdb4 = np.zeros((nL4,1))





# Forward
def forward(W1,W2,W3,W4, b1,b2,b3,b4, X):
    
    Z1 = np.dot(W1, X) + b1
    A1 = activation("relu", Z1)
    A1 = normalize(A1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = activation("relu", Z2)
    A2 = normalize(A2)
    
    Z3 = np.dot(W3, A2) + b3
    A3 = activation("relu", Z3)
    A3 = normalize(A3)
    
    Z4 = np.dot(W4, A3) + b4
    A4 = activation("sigmoid", Z4)
    
    Y_hat = A4
    return Z1,Z2,Z3,Z4, A1,A2,A3,A4,Y_hat




# Cost
def cost(Y_hat,Y):
  Dr = np.shape(Y)[0] * np.shape(Y)[1]
  J =  (1/Dr) * (sum(sum(np.square(Y-Y_hat))))
  return J


# Mean Square Error
def MeanSE(Y_hat,Y):
  Dr = np.shape(Y)[0] * np.shape(Y)[1]
  MSE = (1/Dr) * (sum(sum(np.square(Y-Y_hat))))
  return MSE

# Adam Optimizer
def Adam1(dW, db, VdW, Vdb, SdW, Sdb, beta1, beta2, epsilon, t):
  
  VdW = beta1*VdW + (1-beta1)*dW
  Vdb = beta1*Vdb + (1-beta1)*db
  SdW = beta2*SdW + (1-beta2)*dW**2
  Sdb = beta2*Sdb + (1-beta2)*db**2

  VdW_corrected = VdW/(1-beta1**t)
  Vdb_corrected = Vdb/(1-beta1**t)

  SdW_corrected = SdW/(1-(beta2**t))
  Sdb_corrected = Sdb/(1-(beta2**t))

  W_Adam = VdW_corrected/(np.sqrt(SdW_corrected)+epsilon)
  b_Adam = Vdb_corrected/(np.sqrt(Sdb_corrected)+epsilon)

  return W_Adam, b_Adam, VdW, Vdb,SdW, Sdb


# Gradient with Adam
def Gradient_Adam(X, Y, Y_hat, W1,W2,W3,W4, A1,A2,A3 ,Z1,Z2,Z3,Z4, VdW1, Vdb1, SdW1, Sdb1, VdW2, Vdb2, SdW2, Sdb2, VdW3, Vdb3, SdW3, Sdb3, VdW4, Vdb4, SdW4, Sdb4,  beta1, beta2, epsilon, t):
    dY_hat = Y_hat - Y
    dZ4 = dY_hat * diff("sigmoid", Z4)
    dW4 = np.dot(dZ4, np.transpose(A3)) / m
    db4 = np.sum(dZ4, axis=1) / m
    db4 = np.reshape(db4, (np.shape(db4)[0],1))
    W_Adam4, b_Adam4, VdW4, Vdb4, SdW4, Sdb4 = Adam1(dW4, db4, VdW4, Vdb4, SdW4, Sdb4, beta1, beta2, epsilon, t)

    dA3 = np.dot(np.transpose(W4), dZ4)
    dZ3 = dA3 * diff("relu", Z3)
    dW3 = np.dot(dZ3, np.transpose(A2)) / m
    db3 = np.sum(dZ3, axis=1) / m
    db3 = np.reshape(db3, (np.shape(db3)[0],1))
    W_Adam3, b_Adam3, VdW3, Vdb3, SdW3, Sdb3 = Adam1(dW3, db3, VdW3, Vdb3, SdW3, Sdb3, beta1, beta2, epsilon, t)


    dA2 = np.dot(np.transpose(W3), dZ3)
    dZ2 = dA2 * diff("relu", Z2)
    dW2 = np.dot(dZ2, np.transpose(A1)) / m
    db2 = np.sum(dZ2, axis=1) / m
    db2 = np.reshape(db2, (np.shape(db2)[0],1))
    W_Adam2, b_Adam2, VdW2, Vdb2,SdW2, Sdb2 = Adam1(dW2, db2, VdW2, Vdb2, SdW2, Sdb2, beta1, beta2, epsilon, t)
    
    
    dA1 = np.dot(np.transpose(W2), dZ2)
    dZ1 = dA1 * diff("relu", Z1)
    dW1 = np.dot(dZ1, np.transpose(X)) / m
    db1 = np.sum(dZ1, axis=1) / m
    db1 = np.reshape(db1, (np.shape(db1)[0],1))
    W_Adam1, b_Adam1, VdW1, Vdb1,SdW1, Sdb1 = Adam1(dW1, db1, VdW1, Vdb1, SdW1, Sdb1, beta1, beta2, epsilon, t)
    
    
    return dW1,dW2,dW3,dW4, db1,db2,db3,db4, W_Adam1, b_Adam1, W_Adam2, b_Adam2, W_Adam3, b_Adam3, W_Adam4, b_Adam4, VdW4, Vdb4, SdW4, Sdb4, VdW3, Vdb3, SdW3, Sdb3, VdW2, Vdb2,SdW2, Sdb2, VdW1, Vdb1,SdW1, Sdb1


# Update
def update_Adam(W1,W2,W3,W4, b1,b2,b3,b4, W_Adam1, b_Adam1, W_Adam2, b_Adam2, W_Adam3, b_Adam3, W_Adam4, b_Adam4, alpha):
    W1 = W1 - alpha*W_Adam1
    b1 = b1 - alpha*b_Adam1
    
    W2 = W2 - alpha*W_Adam2
    b2 = b2 - alpha*b_Adam2
    
    W3 = W3 - alpha*W_Adam3
    b3 = b3 - alpha*b_Adam3

    W4 = W4 - alpha*W_Adam4
    b4 = b4 - alpha*b_Adam4

    
    return W1,W2,W3,W4, b1,b2,b3,b4

# Normalize data
def normalize(A):
    A=(A-np.mean(A))/np.std(A)
    return A


# Mini Batch
def mini_Batch(m, mini_Size, X, Y, W1,W2,W3,W4, b1,b2,b3,b4, VdW1, Vdb1, SdW1, Sdb1, VdW2, Vdb2, SdW2, Sdb2, VdW3, Vdb3, SdW3, Sdb3, VdW4, Vdb4, SdW4, Sdb4, alpha,  beta1, beta2, epsilon, t):
  Total_mini = int(np.floor(m/mini_Size))
  J = []
  MSE = []

  for i in range(Total_mini):
    X_mini = X[:, i*mini_Size:(i+1)*mini_Size]
    Y_mini = Y[:, i*mini_Size:(i+1)*mini_Size]

    Z1,Z2,Z3,Z4, A1,A2,A3,A4,Y_hat = forward(W1,W2,W3,W4, b1,b2,b3,b4, X_mini)
    
    
    J_mini = cost(Y_hat,Y_mini)
    MSE_mini = MeanSE(Y_hat,Y_mini)

    dW1,dW2,dW3,dW4, db1,db2,db3,db4, W_Adam1, b_Adam1, W_Adam2, b_Adam2, W_Adam3, b_Adam3, W_Adam4, b_Adam4, VdW4, Vdb4, SdW4, Sdb4, VdW3, Vdb3, SdW3, Sdb3, VdW2, Vdb2,SdW2, Sdb2, VdW1, Vdb1,SdW1, Sdb1 = Gradient_Adam(X_mini, Y_mini, Y_hat, W1,W2,W3,W4, A1,A2,A3 ,Z1,Z2,Z3,Z4, VdW1, Vdb1, SdW1, Sdb1, VdW2, Vdb2, SdW2, Sdb2, VdW3, Vdb3, SdW3, Sdb3, VdW4, Vdb4, SdW4, Sdb4,  beta1, beta2, epsilon, t)
    
    W1,W2,W3,W4, b1,b2,b3,b4 = update_Adam(W1,W2,W3,W4, b1,b2,b3,b4, W_Adam1, b_Adam1, W_Adam2, b_Adam2, W_Adam3, b_Adam3, W_Adam4, b_Adam4, alpha)
    J.append(J_mini)
    MSE.append(MSE_mini)

  if m%mini_Size == 0:
    J = sum(J)/Total_mini
    MSE = sum(MSE)/Total_mini

    
  if m%mini_Size != 0:
    X_mini = X[:, mini_Size*Total_mini:]
    Y_mini = Y[:, mini_Size*Total_mini:]
    
    Z1,Z2,Z3,Z4, A1,A2,A3,A4,Y_hat = forward(W1,W2,W3,W4, b1,b2,b3,b4, X_mini)

    J_mini = cost(Y_hat,Y_mini)
    MSE_mini = MeanSE(Y_hat,Y_mini)
    dW1,dW2,dW3,dW4, db1,db2,db3,db4, W_Adam1, b_Adam1, W_Adam2, b_Adam2, W_Adam3, b_Adam3, W_Adam4, b_Adam4, VdW4, Vdb4, SdW4, Sdb4, VdW3, Vdb3, SdW3, Sdb3, VdW2, Vdb2,SdW2, Sdb2, VdW1, Vdb1,SdW1, Sdb1 = Gradient_Adam(X_mini, Y_mini, Y_hat, W1,W2,W3,W4, A1,A2,A3 ,Z1,Z2,Z3,Z4, VdW1, Vdb1, SdW1, Sdb1, VdW2, Vdb2, SdW2, Sdb2, VdW3, Vdb3, SdW3, Sdb3, VdW4, Vdb4, SdW4, Sdb4,  beta1, beta2, epsilon, t)
    
    W1,W2,W3,W4, b1,b2,b3,b4 = update_Adam(W1,W2,W3,W4, b1,b2,b3,b4, W_Adam1, b_Adam1, W_Adam2, b_Adam2, W_Adam3, b_Adam3, W_Adam4, b_Adam4, alpha)
    J.append(J_mini)
    MSE.append(MSE_mini)

    J = sum(J)/(Total_mini+1)
    MSE = sum(MSE)/(Total_mini+1)

  

  return W1,W2,W3,W4, b1,b2,b3,b4, J, MSE, Y_hat

