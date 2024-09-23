from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from data.data_loader import DataModule
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import plotly.graph_objects as go
import logging
import os
import datetime

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
logging.info('App started')

# setup scheduler
scheduler = BackgroundScheduler()

# data variable to store the data
dataModule = DataModule()
scheduler.add_job(dataModule.nextData, IntervalTrigger(seconds=120))

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
    if not dataModule.obsolete:
        dataModule.obsolete = True
        return dataModule.data
    else:
        return None

@app.get('/')
def getRoot(request: Request) -> HTMLResponse:
    graph = go.Figure()
    graph.add_trace(go.Scatter(
        x=dataModule.data['dates'],
        y=dataModule.data['real'],
        mode='lines+markers',
        name='Real Data')
    )
    graph.add_trace(
        go.Scatter(
            x=dataModule.data['dates'],
            y=dataModule.data['pred'],
            mode='lines+markers',
            name='Predicted Data')
    )
    # graph.write_image('static/plot.png')

    graph.update_layout(
        xaxis=dict(
            rangeslider=dict(
                visible=True
            ),
            type='date'
        )
    )

    graph_json = graph.to_json()
    logging.info('GET root')

    return templates.TemplateResponse(
        'index.html', 
        {
            'request': request,
            'graph_json': graph_json,
        }
    )