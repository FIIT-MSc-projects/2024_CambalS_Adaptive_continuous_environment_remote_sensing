import plotly.graph_objects as go
import os
import logging
import datetime
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from .data.data_loading import DataModule
from plotly.subplots import make_subplots

# Create an instance of FastAPI
app = FastAPI()

# jinja2 templates and static folder setup
templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

# setup logging
if os.path.exists('logs') == False:
    os.mkdir('logs')
logging.basicConfig(
    filename='logs/' + str(datetime.datetime.now().date()) + '.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.info('App started')

# setup scheduler
scheduler = BackgroundScheduler()

# data variable to store the data
dataModule = DataModule()
scheduler.add_job(dataModule.incrementIndex, IntervalTrigger(seconds=20))

# start the scheduler
scheduler.start()

@app.on_event('shutdown')
def shutdown_event():
    scheduler.shutdown()

@app.get('/fulldata')
def getData2() -> dict:
    return dataModule.data

@app.get('/data')
def getData() -> dict | None:
    if dataModule.obsolete == False:
        dataModule.obsolete = True
        return dataModule.data
    else:
        return None

@app.get('/')
def getRoot(request: Request) -> HTMLResponse:
    graph = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("PM10 particulate matter", "PM2.5 particulate matter", "NO2"))

    # 1st subplot
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM10'],
            mode='lines+markers',
            name='Real Data'
        ), row=1, col=1
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM10pred'],
            mode='lines+markers',
            name='Predicted Data'
        ), row=1, col=1
    )

    # 2nd subplot
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM25'],
            mode='lines+markers',
            name='Real Data'
        ), row=2, col=1
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM25pred'],
            mode='lines+markers',
            name='Predicted Data'
        ), row=2, col=1
    )

    # 3rd subplot
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['NO2'],
            mode='lines+markers',
            name='Real Data'
        ), row=3, col=1
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['NO2pred'],
            mode='lines+markers',
            name='Predicted Data'
        ), row=3, col=1
    )
    # graph.write_image('static/plot.png')

    # graph.update_layout(
    #     xaxis=dict(
    #         rangeslider=dict(
    #             visible=True
    #         ),
    #         type='date'
    #     )
    # )

    graph_json = graph.to_json()
    logging.info('GET root')

    return templates.TemplateResponse(
        'index.html', 
        {
            'request': request,
            'graph_json': graph_json,
        }
    )