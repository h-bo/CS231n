      #########################################################################
      # TODO:                                                                 #
      # Sample batch_size elements from the training data and their           #
      # corresponding labels to use in this round of gradient descent.        #
      # Store the data in X_batch and their corresponding labels in           #
      # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
      # and y_batch should have shape (batch_size,)                           #
      #                                                                       #
      # Hint: Use np.random.choice to generate indices. Sampling with         #
      # replacement is faster than sampling without replacement.              #
      #########################################################################
      indices = np.random.choice(X.shape[0], batch_size, replace=True)
      X_batch = X[indices]
      y_batch = y[indices]
      #########################################################################
      #                       END OF YOUR CODE                                #
      #########################################################################
      
      y_pred = np.argmax(scores, axis = 1)
      
      
      
      
    