import numpy as np
import pandas as pd
import random

def shuffle_data():
    eeg_df = pd.read_csv(r"C:\Users\mcss\Data sets for AI_ML\Epileptic_Seizure_Recognition.csv")
    rows = list(range(eeg_df.shape[0]))
    random.shuffle(rows)
    eeg_df = eeg_df.iloc[rows].reset_index(drop = True)  
    return eeg_df                                                                                                
                                                 
def split_data():
    eeg_df = shuffle_data()
    tr_d = eeg_df.iloc[:7000, 1:180]
    va_d = eeg_df.iloc[7000:8500, 1:]
    tst_d = eeg_df.iloc[8500:, 1:]
    return tr_d, va_d, tst_d

def restructure_data():
    tr_d, va_d, tst_d = split_data()
    training_inputs_raw = tr_d.iloc[:, :178].values
    training_inputs = [np.reshape(x, (178, 1)) for x in training_inputs_raw]  # List of arrays, each array is a column vector (X1, X2, X3.....X178)

    training_results_raw = tr_d.iloc[:, 178].values
    training_results = [vectorized_result(y) for y in training_results_raw] # List of arrays, each array is a column vector  
                                                                            # if y = 3, array (0, 0, 1, 0, 0)
    training_data = list(zip(training_inputs, training_results))
    
    validation_inputs = [np.reshape(x, (178, 1)) for x in va_d.iloc[:, :178].values]
    validation_data = list(zip(validation_inputs, list(va_d.iloc[:, 178].values)))

    testing_inputs = [np.reshape(x, (178, 1)) for x in tst_d.iloc[:, :178].values]
    testing_data = list(zip(testing_inputs, list(tst_d.iloc[:, 178].values)))

    return training_data, validation_data, testing_data

def vectorized_result(y):
    vector = np.zeros((5, 1))
    vector[y - 1] = 1.0
    return vector

def restructure_for_ML():
    training_data, validation_data, testing_data = split_data()
    x_train = training_data.iloc[:, :178].values
    y_train = training_data.iloc[:, 178].values
    x_val = testing_data.iloc[:, :178].values
    y_val = testing_data.iloc[:, 178].values
    return x_train, y_train, x_val, y_val

def scaled_NN():
    x_train, y_train, x_test, y_test = restructure_for_ML()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_trainS = scaler.fit_transform(x_train)
    x_testS = scaler.transform(x_test)
    training_inputs = [np.reshape(x, (178, 1)) for x in x_trainS]
    training_results = [vectorized_result(y) for y in y_train] 
    testing_inputs = [np.reshape(x, (178, 1)) for x in x_testS]
    testing_results = list(y_test)
    training_data = list(zip(training_inputs, training_results))
    testing_data = list(zip(testing_inputs, testing_results))
    return training_data, testing_data

def Binary_Classification_Restructure():
    x_train, y_train, x_test, y_test = restructure_for_ML() 
    Non_seizure = [2, 3, 4, 5]
    for i in range(len(y_train)):
        if y_train[i] in Non_seizure : y_train[i] = 0
    for j in range(len(y_test)):
        if y_test[j] in Non_seizure: y_test[j] = 0
    from imblearn.combine import SMOTEENN                             # SMOTEENN = SMOT + ENN, SMOT generates artificial data for minor class in the dataset 
    smen = SMOTEENN()                                                  # ENN cleans the data set by removing data points which may be incorrectly classified    
    x_train1, y_train1 = smen.fit_resample(x_train, y_train)
    return x_train1, y_train1, x_test, y_test

def vectorized_result_NN(y):
    vector = np.zeros((2, 1))
    vector[y - 1] = 1.0
    return vector

def Binary_Clf_NN():
    x_train, y_train, x_test, y_test = Binary_Classification_Restructure()
    training_inputs = [np.reshape(x, (178, 1)) for x in x_train]
    training_results = [vectorized_result_NN(y) for y in y_train]
    testing_inputs = [np.reshape(x, (178, 1)) for x in x_test]
    testing_results = list(y_test)
    training_data = list(zip(training_inputs, training_results))
    testing_data = list(zip(testing_inputs, testing_results))
    return training_data, testing_data

def Binary_NN_scaled():
    x_train, y_train, x_test, y_test = Binary_Classification_Restructure()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_trainS = scaler.fit_transform(x_train)
    x_testS = scaler.transform(x_test)
    training_inputs = [np.reshape(x, (178, 1)) for x in x_trainS]
    training_results = [vectorized_result_NN(y) for y in y_train]
    testing_inputs = [np.reshape(x, (178, 1)) for x in x_testS]
    testing_results = list(y_test)
    training_data = list(zip(training_inputs, training_results))
    testing_data = list(zip(testing_inputs, testing_results))
    return training_data, testing_data  

def testing():
    training_data, validation_data, testing_data = split_data()
    x_test = validation_data.iloc[:, :178].values
    y_test = validation_data.iloc[:, 178].values
    Non_seizure = [2, 3, 4, 5]
    for i in range(len(y_test)):
        if y_test[i] in Non_seizure : y_test[i] = 0
    return x_test, y_test