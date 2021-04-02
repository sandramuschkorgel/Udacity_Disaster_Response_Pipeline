# Disaster Response Pipeline Project
During natural or man-made disasters support for those in need is diverse, highly individualized, and usually time sensitive. People not only call emergency services directly but increasingly send messages via social media platforms. Analyzing these text messages or tweets via Natural Language Processing (NLP) can help to quickly assess whether a message is disaster-related and if so what kind of disaster it is associated with. Based on this, helpers can be deployed in a targeted manner.

## Description
The Flask-based web application provides a user interface for a text message classifier which is trained on a dataset provided by [FigureEight](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) via [Udacity](https://www.udacity.com/). The goal is to create a Machine Learning (ML) model which uses NLP to categorize 
* whether a text message is related to a disaster and
* what kind of disaster the message is referring to.

The repository is separated into three parts:
* data directory
  * two csv files containing the original datasets
  * `process_data.py` merges both csv files and saves the cleaned data in a SQLite database
* models directory
  * `train_classifier.py` returns a classifier model trained on the cleaned data
  * `classifier.pkl` containing the model and its parameters will be stored here (not part of this repository)
* app directory containing all files required to run the web app

## Dependencies/Pre-requisites

## Installation and Usage
```bash
git clone https://github.com/sandramuschkorgel/Udacity_Disaster_Response_Pipeline.git
```

**Cleaning the original datasets** 
```bash
cd [project root directory]
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

**Running the ML Pipeline**
```bash
cd [project root directory]
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

**Using the web interface** 

In order to use the web app it is required to run both, the cleaning step as well as the ML pipeline.

```bash
cd [app sub-directory within project root directory]
python run.py
```

Go to <http://0.0.0.0:3001/>

Use the text input field and type in a message of your choosing. Hit the *Classify Message* button and you are presented with the labels associated with your message.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)