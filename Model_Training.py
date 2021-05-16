mini_Size = 2**9
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
t = 2

MSE_eval = [];
for epoch in range(50):
  
  alpha = 0.00001*np.exp(-0.002*epoch)

  W1,W2,W3,W4, b1,b2,b3,b4, J, MSE, Y_hat = mini_Batch(m, mini_Size, X, Y, W1,W2,W3,W4, b1,b2,b3,b4, VdW1, Vdb1, SdW1, Sdb1, VdW2, Vdb2, SdW2, Sdb2, VdW3, Vdb3, SdW3, Sdb3, VdW4, Vdb4, SdW4, Sdb4, alpha,  beta1, beta2, epsilon, t)
  
  if epoch%5==0:
    print("epoch =", epoch, " "*8 ,"Cost =", MSE)    
    MSE_eval.append(MSE);

print("epoch =", epoch, " "*8 ,"Cost =", MSE)


plot.plot(MSE_eval)
plot.xlabel('Iterations')
plot.ylabel('MSE')
plot.show()
