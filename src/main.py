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
from plotly.subplots import make_subplots
from .data.data_loading import DataModule

# Create an instance of FastAPI
app = FastAPI()

# jinja2 templates and static folder setup
templates = Jinja2Templates(directory='templates')
app.mount("/static", StaticFiles(directory="static"), name="static")

# setup logging
if not os.path.exists('logs'):
    os.mkdir('logs')
logging.basicConfig(
    filename='logs/' + str(datetime.datetime.now().date()) + '.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.info('APP STARTED')

# setup scheduler
scheduler = BackgroundScheduler()

# data variable to store the data
dataModule = DataModule()

# add job to the scheduler
scheduler.add_job(
    dataModule.incrementIndex,
    IntervalTrigger(seconds=10)
)

# start the scheduler
scheduler.start()


@app.on_event('shutdown')
def shutdown_event():
    scheduler.shutdown()
    logging.info('APP SHUTDOWN')


@app.get('/fulldata')
def getFulLData() -> dict:
    logging.info('GET /fulldata')
    return dataModule.data


@app.get('/data')
def getData() -> dict | None:
    if not dataModule.obsolete:
        dataModule.obsolete = True
        return dataModule.data
    else:
        return None


@app.get('/')
def getRoot(request: Request) -> HTMLResponse:
    graph = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "PM10 particulate matter",
            "PM2.5 particulate matter",
            "NO2"
        ),
        vertical_spacing=0.05
    )

    # 1st subplot
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM10'],
            mode='lines+markers',
            name='PM10 real measurement',
            line=dict(color='orange'),
            marker=dict(color='orange'),
            showlegend=True,
            legendgroup='PM10'
        ), row=1, col=1
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM10pred'],
            mode='lines+markers',
            name='PM10 prediction',
            line=dict(color='cyan'),
            marker=dict(color='cyan'),
            showlegend=True,
            legendgroup='PM10'
        ), row=1, col=1
    )

    # 2nd subplot
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM25'],
            mode='lines+markers',
            name='PM2.5 real measurement',
            line=dict(color='orange'),
            marker=dict(color='orange'),
            showlegend=True,
            legendgroup='PM25'
        ), row=2, col=1
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['PM25pred'],
            mode='lines+markers',
            name='PM2.5 prediction',
            line=dict(color='cyan'),
            marker=dict(color='cyan'),
            showlegend=True,
            legendgroup='PM25'
        ), row=2, col=1
    )

    # 3rd subplot
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['NO2'],
            mode='lines+markers',
            name='NO2 real measurement',
            line=dict(color='orange'),
            marker=dict(color='orange'),
            showlegend=True,
            legendgroup='NO2'
        ), row=3, col=1
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['NO2pred'],
            mode='lines+markers',
            name='NO2 prediction',
            line=dict(color='cyan'),
            marker=dict(color='cyan'),
            showlegend=True,
            legendgroup='NO2'
        ), row=3, col=1
    )
    # graph.write_image('static/plot.png')

    graph.update_layout(
        height=840,  # Adjust the height here
        plot_bgcolor='rgba(30, 30, 30, 0.7)',
        paper_bgcolor='rgba(71, 71, 71, 1)',
        font=dict(color='white'),
        title_font=dict(color='white'),
        xaxis3=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05
            ),
            type='date'
        )
    )

    graph_json = graph.to_json()
    logging.info('GET /')

    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'graph_json': graph_json,
        }
    )
