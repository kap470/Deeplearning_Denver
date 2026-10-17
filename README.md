# Introduction to Deep Learning Class Project
```
Team Deeplearning_Denver
Cecilia Hunter
Kevin Perez
```
## Run exploratory analysis
```
There are three scripts to run the exploratory analysis of the data locatend in .src:
precip_temp_IDE.py
Streamflow_IDE.py
gw_analysis_IDE.py
Run these scripts to get Figures and csv files with data useful to get statistics.
```
## Data preprocessing
```
In the folder .src run:
-Reshape_wells.py to convert the gw raw data located in ../data/gw_levels.csv into separate time series,
but also to remove wells with more thatn 40% of missing data.
-preprocess_input.py to generate the file ../data/input_model.csv that contains uniform input series to train the 
Machine Learning and Deep Learning models.
```