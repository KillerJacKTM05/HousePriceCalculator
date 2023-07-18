import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

housing_db_analysis = pd.read_csv('housing_reorganized.csv')
#housing_db_analysis.info()
valueAxis = housing_db_analysis[["housing_median_age", "median_income","bedroom_per_house","total_rooms_per_house","population_per_house","coordinates","ocean_proximity__1h_ocean",
                        "ocean_proximity_inland","ocean_proximity_island","ocean_proximity_near_bay","ocean_proximity_near_ocean"]]
predictionAxis = housing_db_analysis[["median_house_value"]]
#using the standard of 80% train, 20% test ratio.
valueTrain, valueTest, predictionTrain, predictionTest = train_test_split(valueAxis, predictionAxis,random_state=42, shuffle=True, test_size=0.2)
#start training
model = XGBRegressor()
model.fit(valueTrain, predictionTrain)
model_predictions = model.predict(valueTest)

#print(predictionTest.shape)
#print(valueTest.shape)
#predictionTest is a dataFrame from pandas, where model_predictions is a numpy array. First solve this
#converting the dataframe to the 1D series.
prediction_frame = pd.DataFrame({'Actual Value:': predictionTest['median_house_value'].reset_index(drop=True), 'Predicted Value': model_predictions})
#prediction_frame.info()

#Now it's the plotting phase.
print(prediction_frame)
prediction_frame.to_csv('prediction.csv')
predictionFigure = plt.figure(figsize = (9,9))
prediction_frame = prediction_frame.reset_index()
prediction_frame = prediction_frame.drop(['index'], axis=1)
plt.plot(prediction_frame[:100])
plt.legend(['Actual Value', 'Predicted Value'])
plt.savefig("OutputFigure.png")

#Evaluation Part will be added.
