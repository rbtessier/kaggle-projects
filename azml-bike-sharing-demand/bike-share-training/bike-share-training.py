# Import libraries
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def when_is_it(hour):
    if hour >=5 and hour < 11:
        return 'morning'
    elif hour >=11 and hour < 16:
        return 'afternoon'
    elif hour >=16 and hour < 18:
        return 'rush_hour'
    else:
        return 'off_hours'

def season_is_it(season_int):
    if season_int == 1:
        return 'spring'
    elif season_int == 2:
        return 'summer'
    elif season_int == 3:
        return 'fall'
    else:
        return 'winter'
    
def weather_is_it(weather_int):
    if weather_int == 1:
        return 'nice'
    elif weather_int == 2:
        return 'misty'
    elif weather_int == 3:
        return 'ugly'
    else:
        return 'stormy'
    
#function takes all nominal features and converts to dummy features
def dummy_conversion(data): 
    #create the rules for each dummy variable
    df_when_is_it = data['when_is_it'].apply(when_is_it)
    df_season_is_it = data['season'].apply(season_is_it)
    df_weather_is_it = data['weather'].apply(weather_is_it)
    
    
    when_dummies = pd.get_dummies(df_when_is_it, prefix = 'when_')    
    season_dummies = pd.get_dummies(df_season_is_it, prefix = 'season_')
    weather_dummies = pd.get_dummies(df_weather_is_it, prefix = 'weather_')
    
    #drop the old nominal veriables
    data=data.drop('datetime', axis = 1 )
    data=data.drop('season', axis = 1)
    data=data.drop('weather', axis = 1)
    
    data[list(when_dummies.columns)] = when_dummies
    data[list(season_dummies.columns)] = season_dummies
    data[list(weather_dummies.columns)] = weather_dummies
    return data       

#imports train.csv    
bikes = pd.read_csv('train.csv')
#imports test data and saves a temporary record of the datetime column



feature_cols = ['datetime', 'season', 'holiday', 'workingday', 'weather', 'temp',
       'atemp', 'humidity', 'windspeed']    

#saves the labels and the featurs of the training data separately
y = bikes['count']
bikes=bikes[feature_cols]

#splites up the datetime into a nicer format
bikes['when_is_it']= bikes['datetime'].apply(lambda x:int(x[11]+x[12]))

                      


#converts datetime, season and weather nominal variables into dummy variable columns
X=dummy_conversion(bikes)



# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('train.csv')

#initiates preprocessing StandardScaler - which is different from MinMaxScaler I assume, that's what I used last time.
#if I decide to use argparser, here's a place I could use it.
scaler = preprocessing.StandardScaler()

#fits scaler to X
scaler.fit(X)

#sets X to normalized values
X=scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)
    #X_train[feature_cols] = scaler.transform(X_train[feature_cols])
    #X_test[feature_cols] = scaler.transform(X_test[feature_cols])
    #X_train and y_train will be used to train the model
    #X_test and y_test will be used to test the model
    #remember that all four of these variables are just subsets of the overall X and Y
    
    
# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Set regularization hyperparameter

# Train a linear regression model
print('Training a linear regression model')
model = LinearRegression().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)


#ensure negative values predict 0, since negatives make no sense here
y_hat[y_hat <0] = 0

y_hat.to_csv('test_prediction')

error = np.sqrt(metrics.mean_squared_log_error(y_test, y_hat))
    #error = np.sqrt(metrics.mean_squared_log_error(y_test, y_pred))
    # calculate our metric

#logs the error, its extra cool to see this on my Run Metrics in the ML Service 
run.log('Root Mean Squared Log Error', np.float(error))


# calculate AUC - not relevant since this is regression, just commenting it out to remember this later
#y_scores = model.predict_proba(X_test)
#auc = roc_auc_score(y_test,y_scores[:,1])
#print('AUC: ' + str(auc))
#run.log('AUC', np.float(auc))

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/bike_share_model_StandardScale_LinRegression.pkl')

run.complete()
