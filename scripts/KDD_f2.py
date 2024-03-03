#functions related to model training

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import matplotlib.pyplot as plt
import pandas as pd




class Model:
    def __init__(self):
        self.name       = None
        self.model      = None
        self.params     = None
        self.base_eval  = None
        self.best_eval  = None

        

model_obj = Model()


#Modeling scripts: These files contain code that trains and evaluates machine learning models, such as classification, regression, or clustering models.
# They often take the preprocessed data files as input and output the trained models and their evaluation metrics.

# Input: 1- clean_data: Cleaned data frame, a data frame that has been checked for missing values, ourliers, normilzed data, being blanced and unbiased, being randomly suffled, reduce dimention by reducing dependent attributes
#        2- target_attribute: the attribute that we are going to determine by our model through the rest of features and attributes

def data_split(data, target_attribute):
    # seperateing target attribute from dataframe
    #instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt

    # ------------------------------------------------------------------


    data = pd.get_dummies(data, columns = ['season','weathersit','mnth','weekday'],drop_first=True)

    data['cnt_lag_1'] = data['cnt'].shift(-1)
    data['cnt_lag_2'] = data['cnt'].shift(-2)
    data = data.dropna()




    target = [target_attribute]
    features = list(data.columns)
    features.remove('instant')
    features.remove('dteday')
    features.remove('cnt')

    X = data[features]
    y = data[target]

    # spliting Training and Testing Data
    test_ratio = 0.2  # test_data/total_data
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=test_ratio, random_state=1)
    
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    print("number of instances:  ", X_train.shape[0])
    print("          attributes: ", X_train.shape[1]+1 )

    

    return X_train, y_train, X_test, y_test


def model_opt(model, param_grid, X_train, y_train):
    # Initiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='r2', cv=5, n_jobs=5, verbose=2)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Retrieve the best parameters
    best_params = grid_search.best_params_

    # Retrieve the best model
    best_grid = grid_search.best_estimator_

    return best_grid, best_params


def evaluate_model(model, X_test_original, y_test_original, X_train_model):
    
    # Predict Test Data on original data
    features = list(X_train_model.columns)
    X_test_original = X_test_original[features]
    #print(X_test_original.columns.tolist()[0])

    y_pred = model.predict(X_test_original)

    # Calculate evaluation metrics
    rmse = mean_squared_error(y_test_original, y_pred, squared=False)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    print(' ')
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R^2 Score:", r2)
    print(' ')
    return {'rmse': rmse, 'mae': mae, 'r2': r2 }


# Models:

def linearRegression(X_train, y_train, X_test, y_test):
    model_obj.name = 'linearRegression'

    model = LinearRegression()
    model.fit(X_train, y_train)

    model_obj.model = model
    model_obj.params = None

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test, y_test, X_train)
    model_obj.best_eval = model_obj.base_eval

    return model_obj



def polynomialRegression(X_train, y_train, X_test, y_test, degree):
    model_obj.name = 'polynomialRegression'
    param_grid = {'degree': degree}

    polynomial_features = PolynomialFeatures(degree=degree)
    X_train_poly = polynomial_features.fit_transform(X_train)
    features = list(X_train.columns)
    X_test = X_test[features]
    X_test_poly = polynomial_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    model_obj.model, model_obj.params = model_opt(model, param_grid, X_train_poly, y_train)

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test_poly, y_test, X_train_poly)
    model_obj.best_eval = evaluate_model(model_obj.model, X_test_poly, y_test, X_train_poly)

    return model_obj



def decisionTrees(X_train, y_train, X_test, y_test):
    model_obj.name = 'decisionTrees'
    param_grid = {'max_depth': [5, 10]}
    #param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}


    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    model_obj.model, model_obj.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test, y_test, X_train)
    model_obj.best_eval = evaluate_model(model_obj.model, X_test, y_test, X_train)

    return model_obj



def randomForest(X_train, y_train, X_test, y_test):
    model_obj.name = 'randomForest'
    param_grid = {'max_depth': [5, 10], 'n_estimators': [500, 1000, 2000]}

    model=RandomForestRegressor(max_depth=5, n_estimators =500, random_state=42)
    model.fit(X_train, y_train)


    model_obj.model, model_obj.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test, y_test, X_train)
    model_obj.best_eval = evaluate_model(model_obj.model, X_test, y_test, X_train)

    return model_obj



def supportVectorRegression(X_train, y_train, X_test, y_test):
    model_obj.name = 'supportVectorRegression'
    param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 1]}
    #param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto'], 'C': [1, 10, 100, 1000]}
    
    model = SVR()
    model.fit(X_train, y_train)

    model_obj.model, model_obj.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test, y_test, X_train)
    model_obj.best_eval = evaluate_model(model_obj.model, X_test, y_test, X_train)

    return model_obj



def gradientBoosting(X_train, y_train, X_test, y_test):
    model_obj.name = 'gradientBoosting'
    #param_grid = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200, 500]}
    param_grid = {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 7], 'min_samples_leaf': [1, 3, 5], 'max_features': [2, 3, 4]}


    model = GradientBoostingRegressor(max_depth=3, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_obj.model, model_obj.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test, y_test, X_train)
    model_obj.best_eval = evaluate_model(model_obj.model, X_test, y_test, X_train)

    return model_obj



def neuralNetworks(X_train, y_train, X_test, y_test):
    model_obj.name = 'neuralNetworks'
    #param_grid = {'hidden_layer_sizes': [(100,), (100, 50)], 'activation': ['relu', 'tanh']}
    param_grid = {'hidden_layer_sizes': [(64,64,64), (64,32,32)],
                  'activation': ['relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05],
                  'learning_rate': ['constant','adaptive'], 'max_iter': [10000]}



    model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    model_obj.model, model_obj.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    model_obj.base_eval = evaluate_model(model, X_test, y_test, X_train)
    model_obj.best_eval = evaluate_model(model_obj.model, X_test, y_test, X_train)

    return model_obj




def dataSnapShot(model_obj0, model_data_list, data_name):

    model_name      = model_obj0.name
    model           = model_obj0.model
    model_params    = model_obj0.params
    model_base_eval = model_obj0.base_eval
    model_best_eval = model_obj0.best_eval

    model_data_list.append([model_name, model, model_params, model_base_eval, model_best_eval])
    
    file_name0 = '../models/other/'+ model_name +'_'+ data_name   # If using python: './models/other/'
    file_name =  file_name0 +'.pkl'
    joblib.dump(model, file_name)


    temp = [model_name, model, model_params, model_base_eval, model_best_eval]
    file_name = file_name0 +'.txt'
    with open(file_name, 'w') as file:
        content = ''
        for el in temp :
            content += str(el) + ', '
        file.write(content)

    return model_data_list


def modelSelector(data, data_name, target_attribute):
    model_data_list = []
    X_train, y_train, X_test, y_test = data_split(data, target_attribute)

    model_obj =  linearRegression(X_train, y_train, X_test, y_test)
    model_data_list = dataSnapShot(model_obj, model_data_list, data_name)

    model_obj =  decisionTrees(X_train, y_train, X_test, y_test)
    model_data_list = dataSnapShot(model_obj, model_data_list, data_name)

    model_obj =  randomForest(X_train, y_train, X_test, y_test)
    model_data_list = dataSnapShot(model_obj, model_data_list, data_name)

    model_obj =  supportVectorRegression(X_train, y_train, X_test, y_test)
    model_data_list = dataSnapShot(model_obj, model_data_list, data_name)

    model_obj =  gradientBoosting(X_train, y_train, X_test, y_test)
    model_data_list = dataSnapShot(model_obj, model_data_list, data_name)


    final_models = []
    parameter = 'r2'
    previous = 0.00
    # loop for data sets ...
    for model_data in model_data_list:
        current = model_data[4][parameter]
        #print('current: ', current)
        if current > previous:
            chosen_model_name       = model_data[0]
            chosen_model            = model_data[1]
            chosen_model_params     = model_data[2]
            chosen_model_base_eval  = model_data[3]
            chosen_model_best_eval  = model_data[4]
            previous                = model_data[4][parameter]    
    final_models= [chosen_model_name , chosen_model, chosen_model_params, chosen_model_base_eval, chosen_model_best_eval]
    
    # save final models
    file_name0 = '../models/'+ str(chosen_model_name)+'_'+str(data_name)   # If using python: './models/other/'
    file_name = file_name0 +'.pkl'
    joblib.dump(chosen_model, file_name)
    #print()
    #print('final_models: ', final_models)
    #print('model_list: ', model_data_list)


    file_name = file_name0 +'.txt'
    with open(file_name, 'w') as file:
        content = ''
        for el in final_models :
            content += str(el) + ', '        
        file.write(content)

    return final_models, model_data_list
