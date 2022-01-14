# Disaster-Response-Pipelines

## Overview
The project's goal is to build a model for an AI that can classify disaster messages. An emergency worker can use the web app to enter a new message and receive classification results in numerous categories, allowing them to determine what kind of assistance is required.



![](https://video.udacity-data.com/topher/2018/September/5b967cda_disaster-response-project2/disaster-response-project2.png)

## Project Components

There are three components I needed to complete this project.<br><br>

1. ETL Pipeline<br><br>
    In a Python script, process_data.py, I wrote data cleaning pipeline that:<br>

        Loads the messages and categories datasets<br>
        Merges the two datasets<br>
        Cleans the data<br>
        Stores it in a SQLite database<br>

2. ML Pipeline<br><br>
    In a Python script, train_classifier.py,I wrote a machine learning pipeline that:<br>
    
        Loads data from the SQLite database<br>
        Splits the dataset into training and test sets<br>
        Builds a text processing and machine learning pipeline<br>
        Trains and tunes a model using GridSearchCV<br>
        Outputs results on the test set<br>
        Exports the final model as a pickle file<br>

3. Flask Web App<br><br>
    I used html, css and javascript. For this part, I have done:<br>

        Modify file paths for database and model.<br>
        I added two data visualizations using Plotly in the web app.<br>
