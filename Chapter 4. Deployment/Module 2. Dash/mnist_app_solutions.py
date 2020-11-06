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


cnn = torch.load('mnist_cnn.pth')
cnn.eval()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("MNIST Image Detector"),
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

@app.callback([Output('input-image', 'src'), Output('predicted-digit-text', 'children'), \
                Output('predicted-digit-image', 'src'), Output('certainty', 'figure')],
              [Input('upload', 'contents')])
def update_output(image_str):
    if image_str:
        input_img = string2img(image_str)
        info_str = ''
    else:
        input_img = np.array(Image.open("images/default.jpg"))
        image_str = img2string("images/default.jpg")
        info_str= ' (Default example)'

    input_tensor = torch.tensor(input_img).view(1, 1, *input_img.shape)
    digit = cnn(input_tensor.type(torch.FloatTensor)) # argmax of logits
    res = "*The digit in the image is {}!*".format(digit) + info_str

    # Creating plotly graph
    _,probs = cnn.forward(input_tensor.type(torch.FloatTensor))
    probs = probs.flatten().detach().numpy()
    probs_dict = {'digits':list(range(0, 10)), 'probs':probs}
    fig = px.bar(pd.DataFrame(probs_dict), x='digits', y='probs', title='Prediction Certainty', template='plotly_dark')

    return image_str, res, img2string("images/{}.png".format(digit)), fig

if __name__ == '__main__':
    app.run_server(debug=True)