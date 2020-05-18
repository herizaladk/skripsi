import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.models import Sequential
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

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.set_axis(df['Date'], inplace=True)
        df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

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

        model = load_model('lstm.h5')
        metrics_names = "MSE"

        #Evaluation
        scores = model.evaluate(train_generator, verbose=0)
        eval = ("%s: %.2f%%" % (metrics_names, scores*100))

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

        # Multi Step Training

        def predict(num_prediction, model):
            prediction_list = close_data[-look_back:]

            for _ in range(num_prediction):
                x = prediction_list[-look_back:]
                x = x.reshape((1, look_back, 1))
                out = model.predict(x)[0][0]
                prediction_list = np.append(prediction_list, out)
            prediction_list = prediction_list[look_back-1:]

            return prediction_list

        def predict_dates(num_prediction):
            last_date = df['Date'].values[-1]
            prediction_dates = pd.date_range(
                last_date, periods=num_prediction+1).tolist()
            return prediction_dates

        pred = request.form['num_prediction']
        num_prediction = int(pred)
        forecast = predict(num_prediction, model)
        forecast_dates = predict_dates(num_prediction)
        forecast = forecast.reshape((-1, 1))

        forecast = scaler.inverse_transform(forecast)

        forecast = forecast.reshape((-1))

        close_data = scaler.inverse_transform(close_data)
        close_data = close_data.reshape((-1))

        # Multi Step Chart
        trace1 = go.Scatter(
            x=df['Date'].tolist(),
            y=close_data,
            mode='lines',
            name='Data'
        )
        trace2 = go.Scatter(
            x=forecast_dates,
            y=forecast,
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

        return render_template('results.html', shape=df.shape, name=request.files['file'].filename,eval=eval, graph=graph, graphs=graphs)
    return render_template('results.html')
