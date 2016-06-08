# random.choice 
indices = np.random.choice(X.shape[0], batch_size, replace=True)
      X_batch = X[indices]
      y_batch = y[indices]