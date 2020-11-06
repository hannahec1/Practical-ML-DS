import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import torch
import pandas as pd
from PIL import Image
import numpy as np
from utils import ConvNet, img2string, string2img
import sys
sys.path.append('..')


## Load in the torch model trained on the MNIST dataset

# Initialise app
app = dash.Dash(__name__)

# Create layout component
app.layout = html.Div([
    ## Add a title to our app layout
    dcc.Upload(
        id='upload',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.Div(id='predicted-digit-text', style={'position':'relative', 'left':'40%'}),
    # Images to be displayed
    html.Div([
        html.Div([
            html.H3('Input Image'),
            html.Img(id='input-image', style={'height':'100px', 'width':'100px'}),
        ], className="two columns"),

        html.Div([
             html.Img(src=img2string('images/arrow.png'), style={'height':'50px', 'width':'50px', 'position':'relative', 'top':'90px'}),
        ], className="two columns"),

        html.Div([
            html.H3('Predicted Digit'),
            # dcc.Graph('predicted-digit-image'),
            html.Img(id='predicted-digit-image', style={'height':'100px', 'width':'60px'})
        ], className="two columns"),
    ], className="row", style={'width':'90%', 'position':'relative', 'left':'30%'}),
     # Graphs to display model information
    html.Div(children=dcc.Graph(id='certainty', style={'width':'50%', 'height':'50%'}),
            style = {'padding':'50px', 'position':'relative', 'left':'23%'}),

])

@app.callback([], ## Add 4 output components with appropriate component ids and component properties,
              []) ## Add input component with appropriate component id and component property
def update_output(image_str):
    if image_str: # if someone inputs an image
        input_img = ## 
    else: # if we want to access default image
        input_img = ##

    # Determining correct digit
    input_tensor = ## create tensor to be taken as input and make it the appropriate shape (1, 1, input_img_shape)
    digit =  ## argmax of logits
    res = ## string that tells user what the predicted digit for the image was

    # Creating plotly graph with certainty of object 
    _,probs = ## check the CNN model to know what to obtain here
    probs = ## convert tensor to 1D numpy array
    ## Plot these metrics of prediction certainty with a plotly barchart

    ## return the four outputs we input to the callback decorator

if __name__ == '__main__':
    app.run_server(debug=True)