import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import tensorflow as tf
import numpy as np
from numpy import array
from keras.models import Sequential
from keras import backend
from keras.layers import LSTM, Dense
from keras.preprocessing import sequence
from keras.models import load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import plotly
import plotly.graph_objects as go
import json

ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# def init():
#     global model

#     model = load_model('model.h5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/results", methods=['GET', 'POST'])
def results():

    if request.method == 'POST':
        # CSV Input
        df = pd.read_csv(request.files.get('file'))

        df['Date'] =  pd.to_datetime(df['Date'], format='%Y/%m/%d',dayfirst=True)
        df = df.sort_values(by=['Date'], ascending=[True])
        df.set_index('Date', inplace=True)

        df = df.resample('D').ffill().reset_index()
        df = df.fillna(method='ffill')
        df.drop(columns=['Open', 'High', 'Low', 'Adj Close','Volume'], inplace=True)

        # Splitting Data
        scaler = MinMaxScaler()

        close_data = df['Close'].values
        close_data = close_data.reshape((-1, 1))
        close_data = scaler.fit_transform(close_data)

        split_percent = 0.80
        split = int(split_percent*len(close_data))

        close_train = close_data[:split]
        close_test = close_data[split:]

        date_train = df['Date'][:split]
        date_test = df['Date'][split:]

        # Load Model
        look_back = 15

        train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)
        test_generator = TimeseriesGenerator(
            close_test, close_test, length=look_back, batch_size=1)
        
        def rmse(test_generator,train_generator):
	        return backend.sqrt(backend.mean(backend.square(test_generator - train_generator), axis=-1))

        model = load_model('model.h5', custom_objects={'rmse': rmse})
        # metrics_names = "RMSE"

        # Training
        close_train = scaler.inverse_transform(close_train)
        close_test = scaler.inverse_transform(close_test)

        prediction = model.predict_generator(test_generator)
        prediction = scaler.inverse_transform(prediction)

        close_train = close_train.reshape((-1))
        close_test = close_test.reshape((-1))
        prediction = prediction.reshape((-1))

        trace1 = go.Scatter(
            x=date_train,
            y=close_train,
            mode='lines',
            name='Data'
        )
        trace2 = go.Scatter(
            x=date_test,
            y=prediction,
            mode='lines',
            name='Prediksi'
        )
        trace3 = go.Scatter(
            x=date_test,
            y=close_test,
            mode='lines',
            name='Target'
        )
        layout = go.Layout(
            title="IHSG | Hasil Training",
            xaxis={'title': "Date"},
            yaxis={'title': "Close"}
        )
        fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

         #Evaluation
        validtest = close_test
        validtest = validtest[:len(prediction)]

        rmspe = (np.sqrt(np.mean(np.square((validtest - prediction) / validtest)))) * 100
        eval =  ("%.2f%%" % ((rmspe)))

        # Multi Step Training

        # def predict(num_prediction, model):
        #     prediction_list = close_data[-look_back:]

        #     for _ in range(num_prediction):
        #         x = prediction_list[-look_back:]
        #         x = x.reshape((1, look_back, 1))
        #         out = model.predict(x)[0][0]
        #         prediction_list = np.append(prediction_list, out)
        #     prediction_list = prediction_list[look_back-1:]

        #     return prediction_list

        # def predict_dates(num_prediction):
        #     last_date = df['Date'].values[-1]
        #     prediction_dates = pd.date_range(
        #         last_date, periods=num_prediction+1).tolist()
        #     return prediction_dates
        pred = request.form['num_prediction']
        num_prediction = int(pred)+1
        close_data = close_data.reshape((-1))
        # num_prediction = pred

        # forecast_list = close_data[-look_back:]
        # for _ in range(num_prediction):
        #     x_input = array(close_data[-look_back:])
        #     x_input = x_input.reshape((1, look_back, 1))
        #     yhat = model.predict(x_input, verbose=0)
        #     forecast_list = np.append(forecast_list,yhat)
        # # print (forecast_list)
        X = train_generator
        y = test_generator

        def split_sequence(sequence, n_steps_in, n_steps_out):
            X, y = list(), list()
            for i in range(len(sequence)):
                # find the end of this pattern
                end_ix = i + n_steps_in
                out_end_ix = end_ix + n_steps_out
                # check if we are beyond the sequence
                if out_end_ix > len(sequence):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
                X.append(seq_x)
                y.append(seq_y)
            return array(X), array(y)

        # define input sequence
        raw_seq = close_data
        # choose a number of time steps
        n_steps_in, n_steps_out = look_back, num_prediction
        # split into samples
        X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))

        model = Sequential()
        model.add(
            LSTM(10,
                activation='relu',
                input_shape=(look_back,1),
                )
        )
        model.add(Dense(num_prediction))
        model.compile(optimizer='Adam', loss='mse')

        model.fit(X, y, epochs=25, verbose=1)

        x_input = raw_seq[-n_steps_in:]
        x_input = x_input.reshape((1, n_steps_in, n_features))
        yhat = model.predict(x_input, verbose=0)


        yhat = yhat.reshape((-1,1))
        yhat = scaler.inverse_transform(yhat)
        yhat = yhat.reshape((-1))

        close_data = close_data.reshape((-1,1))
        close_data = scaler.inverse_transform(close_data)
        close_data = close_data.reshape((-1))

        yhat[:1] = close_data[-1:]
        # print(yhat)

        def predict_dates(num_prediction):
            last_date = df['Date'].values[-1]
            prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
            return prediction_dates

        forecast_dates = predict_dates(num_prediction)

        # pred = request.form['num_prediction']
        # num_prediction = int(pred)+1

        # forecast = predict(num_prediction, model)
        # forecast_dates = predict_dates(num_prediction)
        # forecast = forecast.reshape((-1, 1))

        # forecast = scaler.inverse_transform(forecast)

        # forecast = forecast.reshape((-1))

        # close_data = scaler.inverse_transform(close_data)
        # close_data = close_data.reshape((-1))

        # Multi Step Chart
        trace1 = go.Scatter(
            x=df['Date'].tolist(),
            y=close_data,
            mode='lines',
            name='Data'
        )
        trace2 = go.Scatter(
            x=forecast_dates,
            y=yhat,
            mode='lines',
            name='Prediction'
        )
        layout = go.Layout(
            title="IHSG | Prediksi",
            xaxis={'title': "Date"},
            yaxis={'title': "Close"}
        )

        figs = go.Figure(data=[trace1, trace2], layout=layout)
        graphs = json.dumps(figs, cls=plotly.utils.PlotlyJSONEncoder)

        dftabelcast = pd.Series(yhat)
        dftabeldates = pd.Series(forecast_dates)

        frame = { 'Tanggal': dftabeldates, 'Harga': dftabelcast } 
        result = pd.DataFrame(frame) 
        notes = "Index 0 Adalah Harga Sebelum Prediksi"
        # result.index = np.arange(1,len(result)+1)
        predshow = result.head(num_prediction).to_html(classes = 'frames')

        return render_template('results.html', shape=df.shape, name=request.files['file'].filename,eval=eval, graph=graph, graphs=graphs, predshow = [predshow], titles =['NA'], notes=notes  )
    return render_template('results.html')
