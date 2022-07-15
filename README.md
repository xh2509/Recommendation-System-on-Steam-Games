# E6893_Group45
Recommendations on Steam Games Project

Taotao Jiang(tj2441), Xinyu He(xh2469), Xiaohang He(xh2509)

# Project Description
Our goal of this project is to construct a new system of games recommendation which is influenced by many features. Our system is built based on the data obtained from Steamspy, a website that streams data from Steam. We focus on predicting the self-created features based on raw data for final scorecard of each game. We used Linear Regression, Random Forest Regressor, and Gradient Boosting Regression. A web interface built with JavaScript and HTML5 is used for result visualizaion. The result is futhered by LDA topic analysis.

# Web Interface Link
Url to web interface: https://jsfiddle.net/HXH1620/bL39a2jg/show/

# Data Link
Url to all data used and generated: https://drive.google.com/drive/folders/1Sba1O7oKbyLZxjN3HatpnoF2h3c4RmFh

# Organization of the Files
```
. 
├── data.py  
├── model.py
├── model.ipynb  
├── pro_airflow.py 
├── interface.html
├── interface.js
├── game_scorecard_lda.csv
├── steamspy_data.csv
├── steamspy_onehot.csv
├── steamspy_withlda.csv
├── webdata.csv
├── Final Report.pdf
└── README.md

```

# Description of the Files
* data.py: data collection and preprocessing
* model.py: modelling and LDA
* model.ipynb: the output presentation of models and LDA
* pro_airflow.py: airflow DAG of the project
* interface.html: HTML5 code for web interface
* interface.js: JavaScript code for web interface
* steamspy_data.csv: collected raw data
* steamspy_onehot.csv: data after preprocessing for analysis
* steamspy_withlda.csv: LDA analysis results data
* game_scorecard_lda.csv: complete scorecard of each game including four essential features
* webdata.csv: processsed scorecard data for web interface
* Final Report.pdf: project report
