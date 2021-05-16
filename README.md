1. Input_Output_Layer_Neuron.py 
   Run this .py file 1st.
   Users need to give X and Y as the respective input and output. 
   Users can change the number of neurons as required.
   

2. DNN_Model.py
     Run this after Input_Output_Layer_Neuron.py 
     It has a 4-Layer DNN Structure implemented, with Relu-Relu-Relu-Sigmoid as activation.
     This activation can be changed by the user in the "Forward" and "Gradient with Adam" definitions.
     
3. Model_Training.py
     This file to run 3rd
     It is the training loop.
     Users can Change the Learning rate alpha as required.
     After training, Weights and bias values are modified.
     
4. Evaluation.py
     This is the evaluation code. Run in the last.
     Use this trained Weights and bias with the forward network to evaluate results for the unknown data.
     X_test is the input to the network.
     Y_hat is the output obtain from the DNN Network.     
