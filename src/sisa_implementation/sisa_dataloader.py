import numpy as np
import os

train_data = np.load('src/modelling/data/purchase2_train.npy', allow_pickle=True)
test_data = np.load('src/modelling/data/purchase2_test.npy', allow_pickle=True)

train_data = train_data.reshape((1,))[0]
test_data = test_data.reshape((1,))[0]

X_train = train_data['X'].astype(np.float32)
X_test = test_data['X'].astype(np.float32)
y_train = train_data['y'].astype(np.int64)
y_test = test_data['y'].astype(np.int64)

def load(indices=None, category='train'):
    if category == 'train':
        if indices is None:
            return X_train, y_train
        else:
            return X_train[indices], y_train[indices]
    elif category == 'test':
        if indices is None:
            return X_test, y_test
        else:
            return X_test[indices], y_test[indices]

if __name__ == "__main__":
    # Print shape of X_train and X_test
    print(X_train.shape)
    print(X_test.shape)
