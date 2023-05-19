from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


from joblib import dump, load
import matplotlib.pyplot as plt

best_model = object()



#Modeling scripts: These files contain code that trains and evaluates machine learning models, such as classification, regression, or clustering models.
# They often take the preprocessed data files as input and output the trained models and their evaluation metrics.

# Input: 1- clean_data: Cleaned data frame, a data frame that has been checked for missing values, ourliers, normilzed data, being blanced and unbiased, being randomly suffled, reduce dimention by reducing dependent attributes
#        2- target_attribute: the attribute that we are going to determine by our model through the rest of features and attributes

def data_split(data, target_attribute):
    # seperateing target attribute from dataframe

    feature = data.drop(target_attribute, axis=1)

    # create a isolated data frame for target attribute
    target = data[target_attribute]

    # spliting Training and Testing Data
    test_ratio = 0.2  # test_data/total_data
    X_train, X_test, y_train, y_test = train_test_split(feature, target, shuffle=True, test_size=test_ratio, random_state=1)
    return X_train, y_train, X_test, y_test


def model_opt(model, param_grid, X_train, y_train):
        # Initiate the grid search model
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='recall', cv=5, n_jobs=5, verbose=2)
        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)

        grid_search.best_params_
        ## Evaluating Optimised Model After finding the best parameter for the model 
        # we can access the `best_estimator_` attribute of the GridSearchCV 
        # object to save our optimised model into variable called `best_grid`. 
        # We will calculate the 6 evaluation metrics using our helper function to compare it 
        # with our base model on the next step.
        best_params = grid_search.best_params_
        best_grid = grid_search.best_estimator_

        return best_grid, best_params


def evaluate_model(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    kappa = metrics.cohen_kappa_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'kappa': kappa, 'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm}


# Models:

def linearRegression_(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create the parameter grid based on the results of random search
    param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)

    return best_model


def gaussian(X_train, y_train, X_test, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Evaluate Model
    best_model.model = model
    best_model.params = '-'

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = best_model.base_eval

    return best_model


def kNN(X_train, y_train, X_test, y_test):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)

    # Define the parameter grid
    param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': [1, 2]}

    # Perform grid search on the model using the defined parameter grid
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)


    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)
    

    return best_model


def randomForest(X_train, y_train, X_test, y_test):
    model=RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Create the parameter grid based on the results of random search
    param_grid = {'max_depth': [50, 80, 100], 'max_features': [2, 3, 4], 'min_samples_leaf': [3, 4, 5], 'min_samples_split': [8, 10, 12], 'n_estimators': [100, 300, 500]}
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)

    return best_model


def decisionTree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Create the parameter grid based on the results of random search
    param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)

    return best_model


def neuralNetwork(X_train, y_train, X_test, y_test):
    model = MLPClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Create the parameter grid based on the results of random search





    param_grid = {'hidden_layer_sizes': [(10,), (20,), (50,), (100,), (10,10), (20,20), (50,50), (50,100), (100,50), (100,100), (50,50,50), (50,100,50)],
                  'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.05],
                  'learning_rate': ['constant','adaptive']}
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)

    return best_model


def supportVectorRegression(X_train, y_train, X_test, y_test):
    model = SVR()
    model.fit(X_train, y_train)

    # Create the parameter grid based on the results of random search
    param_grid = {'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                  'gamma': ['scale', 'auto'], 'C': [1, 10, 100, 1000]}
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)

    return best_model


def gradientBoosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=0)
    model.fit(X_train, y_train)
    
    # Create the parameter grid based on the results of random search
    param_grid = {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [50, 100, 200], 'max_depth': [3, 4, 5], 'min_samples_leaf': [1, 3, 5], 'max_features': [2, 3, 4]}
    best_model.model, best_model.params = model_opt(model, param_grid, X_train, y_train)

    # Evaluate Model
    best_model.base_eval = evaluate_model(model, X_test, y_test)
    best_model.best_eval = evaluate_model(best_model.model, X_test, y_test)

    return best_model


def modelSelector(data, target_attribute):
    model_list = []
    X_train, y_train, X_test, y_test = data_split(data, target_attribute)

    best_model =  linearRegression_(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  gaussian(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  kNN(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  randomForest(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  decisionTree(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  neuralNetwork(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  supportVectorRegression(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    best_model =  gradientBoosting(X_train, y_train, X_test, y_test)
    model_list.append(best_model)

    parameter_list = ['acc', 'prec', 'rec', 'f1', 'kappa', 'fpr', 'tpr', 'auc', 'cm']
    models_names = ['LinearRegression_', 'Gaussian', 'KNN', 'RandomForest', 'DecisionTree', 'NeuralNetwork', 'SupportVectorRegression', 'GradientBoosting']
    final_models = []
    
    for parameter in parameter_list:
        for opt_model in model_list:
            if opt_model.best_eval[parameter] > previous:
                chosen_model = opt_model.model
                chosen_model_base_eval = opt_model.base_eval
                chosen_model_best_eval = opt_model.best_eval
        
                previous = opt_model.best_eval[parameter]
        model_index = opt_model.model.index(chosen_model)
        model_name = models_names[model_index]
        final_models.append([parameter, model_name , chosen_model, chosen_model_best_eval, chosen_model_base_eval])
        
        # save final models
        file_name = '../models/' + str(model_name) + '.joblib'
        dump(chosen_model, file_name)

    print(final_models)
    return final_models, model_list