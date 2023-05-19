#Prediction scripts: These files contain code that makes predictions or classifications on new data using the trained models. 
# They often take new data files as input and output the predicted results.


from joblib import load

def prediction(data, target_attribute):
    file_name = '../models/model_name.joblib'
    # Load the saved model
    model = load(file_name)

    # Use the loaded model to make predictions
    X_data = data.drop(target_attribute, axis=1)
    y_pred = model.predict(X_data)

    return y_pred


