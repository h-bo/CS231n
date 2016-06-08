import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i].T
        dW[:, y[i]] -= X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * np.abs(W)

  
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_feature = W.shape[1]
  i = 0
  scores = X.dot(W)
  truescores = np.zeros((num_train, 1))
  
 
  for yi in y:
    truescores[i, 0] = scores[i, yi]
    scores[i, yi] -= 1 # for next decrease dulplicate
    i += 1
  # vertorization version
  # truescores[y, 0] = scores[y]
  # scores[y] -= 1 
  diff = np.zeros(scores.shape)
  diff = scores - truescores + 1 # note delta = 1 
  difftrue = (diff > 0)
  loss = sum(diff[difftrue])
  loss /= num_train
  
  
  # loop version runs correctly
  Xsum = np.sum(difftrue, axis = 1).reshape((-1,1)) * X 
  for i in range(num_train):
    dW[:, y[i]] -= Xsum[i]
    
  # but vertorization version fails
  # so puzzled what's wrong here
  # dW[:, y] -= Xsum.T
  
  difftrue_3d = difftrue.reshape((num_train, 1, num_feature))
  X_3d = X.reshape((X.shape[0], X.shape[1], 1))
  multi_3d = difftrue_3d *  X_3d
  dW += multi_3d.sum(axis = 0)
  
  
  
  
  
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * np.abs(W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
