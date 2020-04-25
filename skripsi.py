import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model
import plotly
import plotly.graph_objects as go
import json

ALLOWED_EXTENSIONS = {'txt', 'csv'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def init():
    global model

    model = load_model('model.h5')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/results", methods=['GET', 'POST'])

def results():

    if request.method == 'POST':
        
        df = pd.read_csv(request.files.get('file'))

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.set_axis(df['Date'], inplace=True)
        df.drop(columns=['Open', 'High', 'Low', 'Volume'], inplace=True)

        trace = go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Data')


        layout = go.Layout(
        title="IHSG | 25 November 2014 - 22 November 2019",
        xaxis={'title': "Date"},
        yaxis={'title': "Close (Rupiah)"}
            )
        fig = go.Figure(data=[trace], layout=layout)
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

          
        return render_template('results.html', shape=df.shape, name = request.files['file'].filename, graph = graph)

    return render_template('results.html')
# Upload file
# Direct New Page
# Load Model
