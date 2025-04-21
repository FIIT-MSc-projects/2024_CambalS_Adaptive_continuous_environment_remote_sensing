import plotly.graph_objects as go
import os
import logging
import datetime
import plotly.express as px
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
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# setup logging
if not os.path.exists("logs"):
    os.mkdir("logs")
logging.basicConfig(
    filename="logs/" + str(datetime.datetime.now().date()) + ".log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logging.info("APP STARTED")

# setup scheduler
scheduler = BackgroundScheduler()

# data variable to store the data
dataModule = DataModule()

# add job to the scheduler
scheduler.add_job(dataModule.incrementIndex, IntervalTrigger(seconds=4))

# start the scheduler
scheduler.start()


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    logging.info("APP SHUTDOWN")


@app.get("/fulldata")
def getFulLData() -> dict:
    logging.info("GET /fulldata")
    return dataModule.data


@app.get("/data")
def getData() -> dict | None:
    if not dataModule.obsolete:
        dataModule.obsolete = True
        return dataModule.data
    else:
        return dataModule.data


@app.get("/predict")
def testEndpoint() -> str:
    logging.info("GET /predict")
    dataModule.test()
    return "Predict endpoint"


@app.get("/")
def getRoot(request: Request) -> HTMLResponse:
    data = dataModule.data
    pollutants = [
        ("PM10", "orange"),
        ("PM25", "orange"),
        ("NO2", "orange"),
    ]
    fig = make_subplots(
        rows=len(pollutants),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[p[0] for p in pollutants],
        vertical_spacing=0.05,
    )

    for i, (gas, color) in enumerate(pollutants, start=1):
        # real
        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data[gas],
                mode="lines+markers",
                name=f"{gas} real",
                line=dict(color=color),
                legendgroup=gas,
            ),
            row=i,
            col=1,
        )
        # prediction
        fig.add_trace(
            go.Scatter(
                x=data["dates"],
                y=data[f"{gas}pred"],
                mode="lines+markers",
                name=f"{gas} pred",
                line=dict(color="cyan"),
                legendgroup=gas,
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        height=840,  # Adjust the height here
        plot_bgcolor="rgba(30, 30, 30, 0.7)",
        paper_bgcolor="rgba(71, 71, 71, 1)",
        font=dict(color="white"),
        title_font=dict(color="white"),
        xaxis3=dict(rangeslider=dict(visible=True, thickness=0.05), type="date"),
    )

    graph_json = fig.to_json()
    logging.info("GET /")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "graph_json": graph_json,
        },
    )
