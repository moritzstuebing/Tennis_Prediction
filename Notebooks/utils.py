import pandas as pd
import numpy as np

def shuffle_data(X, y, seed=0):

    """
        This function shuffles some dataset consisting of a prediction matrix and a target variable that
        are both numpy arrays.
    """
    
    rng = np.random.default_rng(seed)
    indices = np.arange(X.shape[0])
    
    shuffled_indices = rng.permutation(indices)

    return X[shuffled_indices], y[shuffled_indices]

    

def shuffle_data_df(df, seed=0):

    """
        This function shuffles a dataframe
    """
    
    rng = np.random.default_rng(seed)
    indices = np.arange(df.shape[0])
    
    shuffled_indices = rng.permutation(indices)

    return df.iloc[shuffled_indices, :]

def train_test_split(X, y, threshold):

    """
        This function creates a training set and test set for both the prediction and target variables
        that are numpy arrays. It assumes that the data has already been shuffled.
    """

    split_spot = int(threshold * X.shape[0])

   
    X_train, y_train = X[:split_spot, :], y[:split_spot]
    X_test, y_test = X[split_spot:, :], y[split_spot:]


    return X_train, X_test, y_train, y_test

def train_test_split_df(df, threshold):

    """
        This function creates a training set and test set for a dataframe. 
        It assumes that the data has already been shuffled.
    """

    split_spot = int(threshold * df.shape[0])

    # pd dataframe
    df_train = df.iloc[:split_spot, :]
    df_test = df.iloc[split_spot:, :]

    return df_train, df_test

def accuracy(y_pred, y):
    return np.mean(y_pred == y)

def mse(y, y_hat):

    """
        Calculates the mean squared error between a vector of predictions and true values.  
    """

    return np.mean((y - y_hat) ** 2)

def ridge_mse(y, y_hat, lam, theta):

    """
        Given some regularisation parameter and vector of ridge coefficients, 
        calculates the regularised mean squared error between a vector of predictions and true values.
    """

    return np.mean((y - y_hat) ** 2) + lam * (theta.T @ theta)

def make_folds(X_train, k=5):

    """
        Returns fold indices for a dataset for a cross-validation procedure. Assumes the data is
        already shuffled. k gives the number of folds.
    """

    indices = np.arange(X_train.shape[0])
    fold_length = X_train.shape[0] // k
    folds = []

    for i in range(k-1):
        folds.append(indices[fold_length*i : fold_length*(i+1)])

    folds.append(indices[fold_length * (k-1):])

    return folds

def standardise(X, scaler=None):
    if scaler is None:
        X_means = np.mean(X, axis=0)
        X_devs = np.std(X, axis=0)

        return (X - X_means) / X_devs

    else:
        means = np.mean(scaler, axis=0)
        X_devs = np.std(scaler, axis=0)
        
        return (X - means) / X_devs

def standardise_cols(X, ind, scaler=None):

    """
        Standardise columns given by specified indices. Stats come from `scaler`
        if given, else from X itself. Returns a new array; X is left unchanged.
    """

    source = X if scaler is None else scaler
    src = source[:, ind].astype(float)
    col_means = np.mean(src, axis=0)
    col_devs  = np.std(src, axis=0)

    X_out = X.astype(float).copy()                       # work on a copy, don't mutate X
    X_out[:, ind] = (X_out[:, ind] - col_means) / col_devs
    return X_out
