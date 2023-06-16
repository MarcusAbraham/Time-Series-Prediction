import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Visualise the "items bought" values change over time
def visualise_data(file):
    # Read the csv file and slit the first column into three separate columns
    df = pd.read_csv(file, index_col=False)
    df['Year'] = df['Year; Month;ItemsBought'].str.split(';').str[0]
    df['Month'] = df['Year; Month;ItemsBought'].str.split(';').str[1]
    df['ItemsBought'] = df['Year; Month;ItemsBought'].str.split(';').str[2]

    # Convert items bought from strings to floats, and create a combined date column
    df['ItemsBought'] = pd.to_numeric(df['ItemsBought'])
    df['Date'] = df['Month'] + '-' + df['Year']

    # Plot the items bought over time
    plt.plot(df['Date'], df['ItemsBought'])
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Items Bought', fontsize=8)
    plt.title('Monthly Items Bought')
    tick_positions = df.index[::6]
    tick_labels = df['Date'].iloc[::6]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=6)
    plt.yticks(fontsize=8)
    plt.show()


# Create a dataset matrix (look_back = number of time steps to look back at for prediction of the next step)
def create_dataset(dataset, look_back):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


# Create the LSTM model and make predictions
def make_model(file):
    # Set the random seeds for re-reproducibility
    np.random.seed(6)
    tf.random.set_seed(6)

    # Read the csv and create an items bought column
    df = pd.read_csv(file, index_col=False)
    df['ItemsBought'] = df['Year; Month;ItemsBought'].str.split(';').str[2]
    df['Year'] = df['Year; Month;ItemsBought'].str.split(';').str[0]
    df['Month'] = df['Year; Month;ItemsBought'].str.split(';').str[1]
    df['Date'] = df['Month'] + '-' + df['Year']

    dataset = df['ItemsBought'].values.astype(float)

    # Normalise the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    # Split the dataset into training and test data, the test data being 10 time units at the end of the data set
    train_size = int(len(dataset) * 0.935)
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("Training set size: ", len(train), "Testing set size: ", len(test))

    # Reshape the data
    look_back = 1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back), return_sequences=True))
    model.add(LSTM(4))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error')
    model.fit(train_x, train_y, epochs=100, batch_size=1, verbose=2)

    # Make predictions and invert them so that they can be used in calculations
    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])

    # Calculate and print the root mean squared error
    train_score = np.sqrt(mean_squared_error(train_y[0], train_predict[:, 0]))
    print('Train Score: %.2f RMSE' % train_score)
    test_score = np.sqrt(mean_squared_error(test_y[0], test_predict[:, 0]))
    print('Test Score: %.2f RMSE' % test_score)

    # Shift training and test predictions for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict
    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(dataset) - 1, :] = test_predict

    # Plot the original items bought values VS the training and test predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(train_predict_plot)
    plt.plot(test_predict_plot)
    plt.xlabel('Date', fontsize=8)
    plt.ylabel('Items Bought', fontsize=8)
    tick_positions = df.index[::6]
    tick_labels = df['Date'].iloc[::6]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=6)
    plt.show()


visualise_data('buying_history.csv')
make_model('buying_history.csv')
