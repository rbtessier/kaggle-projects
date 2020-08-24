import json
import joblib
import numpy as np
from azureml.core.model import Model


# Called when service is loaded
def init():
    global model
    #ws=Workspace.get(name ='ms-learn-ml', subscription_id='375be063-544d-4b99-aed9-5a0e16b1f428', resource_group='ms-learn-ml')
    #get path to the deployed
    model_path = Model.get_model_path('titanic_model_automl', version = 2)
    model = joblib.load(model_path)
    
#called when a request is received
def run(raw_data):
    
    #Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    #get a predication from the model
    predictions = model.predict(data)
    #get the corresponding classname for each predication (0 or 1)
    classnames = ["died", "survived"]
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
        
    #Return the predictions as JSON
    return json.dumps(predicted_classes)
