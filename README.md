# DP - Bc. Sebastián Čambál

The implementation of the diploma thesis "Detection of air pollution in the Slovak Republic using machine learning methods".

## Project Organization

```txt
.
├── LICENSE
├── README.md
├── data
│   ├── interim
│   │   └── EEA-SK-Ba.csv
│   ├── processed
│   │   ├── EEA-SK-Ba-clean.csv
│   │   ├── EEA-SK-Ba-trend-season.csv
│   │   └── EEA-SK-Ba-trend.csv
│   └── raw
│       └── EEA-SK-Bratislava
│           ├── 2018
│           │   ├── SK_10_27210_2018_timeseries_CO.csv
│           │   ├── SK_1_27236_2018_timeseries_SO2.csv
│           │   ├── SK_20_27277_2018_timeseries_C6H6.csv
│           │   ├── SK_5_27208_2018_timeseries_PM10.csv
│           │   ├── SK_6001_35900_2018_timeseries_PM25.csv
│           │   └── SK_8_27323_2018_timeseries_NO2.csv
│           ├── 2019
│           │   ├── SK_10_27210_2019_timeseries_CO.csv
│           │   ├── SK_1_27236_2019_timeseries_SO2.csv
│           │   ├── SK_20_27277_2019_timeseries_C6H6.csv
│           │   ├── SK_5_27294_2019_timeseries_PM10.csv
│           │   ├── SK_6001_35900_2019_timeseries_PM25.csv
│           │   └── SK_8_27323_2019_timeseries_NO2.csv
│           ├── 2020
│           │   ├── SK_10_27210_2020_timeseries_CO.csv
│           │   ├── SK_1_27236_2020_timeseries_SO2.csv
│           │   ├── SK_20_27277_2020_timeseries_C6H6.csv
│           │   ├── SK_5_27294_2020_timeseries_PM10_1.csv
│           │   ├── SK_6001_35900_2020_timeseries_PM25_1.csv
│           │   └── SK_8_27303_2020_timeseries_NO2_1.csv
│           └── 2021
│               ├── SK_10_27210_2021_timeseries_CO.csv
│               ├── SK_1_68662_2021_timeseries_SO2.csv
│               ├── SK_20_27277_2021_timeseries_C6H6.csv
│               ├── SK_5_27208_2021_timeseries_PM10.csv
│               ├── SK_6001_45549_2021_timeseries_PM25.csv
│               └── SK_8_27323_2021_timeseries_NO2.csv
├── models
│   ├── lstm1.keras
│   ├── lstm2.keras
│   ├── lstm2_retrained.keras
│   ├── lstm_cnn.keras
│   ├── lstm_cnn_skip.keras
│   └── transformer.keras
├── notebooks
│   ├── 0.1-data-preparation.ipynb
│   ├── 0.2-data-exploratory-analysis.ipynb
│   ├── 0.3-data-preprocessing.ipynb
│   ├── 1.0-lstm.ipynb
│   ├── 1.1-transformer.ipynb
│   └── 2.0-data-drift.ipynb
├── requirements.txt
├── setup.py
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   ├── data_loading.py
    │   ├── make_dataset.py
    │   └── model_actions
    │       ├── NN_module.py
    │       ├── anomaly_detection.py
    │       └── drift_detection.py
    ├── features
    │   ├── __init__.py
    │   └── build_features.py
    ├── logs
    │   ├── 2024-10-06.log
    │   ├── ...
    ├── main.py
    ├── random_search
    │   └── ...
    ├── static
    │   ├── EEA-SK-Ba-trend.csv
    │   ├── plot.png
    │   └── plotly-2.35.0.min.js
    └── templates
        └── index.html
```
