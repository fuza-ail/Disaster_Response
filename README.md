# Disaster Response Pipeline Project

### Introduction
The Disaster Response Pipeline project is part of Udacity Data Science Nano Degree project. This project is Natural Language Processing (NLP) pipeline, classifying disaster message into several categories. The project also contains web app to classify the message along with some visualization. The goal of this project is to classify the message into 36 disaster categories.

### ETL Pipeline
The detail of ETL process are:
1. Import all libraries needed such as numpy, pandas, and sqlalchemy.
2. Load messages dataset from csv file.
3. Load categories dataset from csv file.
4. Merge datasets.
5. Split categories into column.
6. Convert categories values into 0 or 1.
7. Remove duplicate values.
8. Save dataset into sqlite database.
  
### ML Pipeline
Machine learning pipeline:
1. Import all python libraries needed.
2. Load dataset from sqlite database.
3. Tokenize the message.
4. Build machine learning pipeline.
5. Train the model.
6. Test the model.
7. Export model as pickle file.


### Usage:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/
