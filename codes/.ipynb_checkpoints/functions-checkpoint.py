import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# Returns a table showing the features that are suspected to be integer

def integer_columns(data):
    
    # Making a copy and dropping the missing values
    data_no_na = data.dropna().copy() 

    num_non_integers = []
    countains_integers = []
    missing_values = []

    for col in data_no_na.select_dtypes(include="float").columns:
        x = data_no_na[col].apply(float.is_integer).all() # Returns a Boolen value , True if the variable only countains integers
        y = data_no_na.shape[0] - data_no_na[col].apply(lambda x : x.is_integer()).sum()
        z = data[col].isna().sum()
    
        num_non_integers.append(y)
        countains_integers.append(x)
        missing_values.append(z)

    d = {"Countains only integers": countains_integers,
         "Number of non-integers" : num_non_integers,
         "Number of missing values" : missing_values}

    is_integer = pd.DataFrame(index = data_no_na.select_dtypes(include="float").columns, data=d)
    integers = is_integer[is_integer["Countains only integers"]==True].copy()
    return integers
    
    
# Returns a table showing the percentage and number of missing values

def percent_miss_values(data, meta_data):
    
    df_percentages = pd.DataFrame(index = ["Sum of missing values", "Percentage"] , columns = data.columns)
    df_percentages.loc["Sum of missing values",:] = data.isna().sum()
    df_percentages.loc["Percentage"]= data.isna().mean()*100

    # We round those values to see them in a more convenient way.
    df_percentages.loc["Percentage"]= df_percentages.loc["Percentage"].astype(float).round(2) 
     

    # We use the transposed method to have a better view on those datas and we have sorted them by descending order
    percent_na = df_percentages.T.sort_values(by="Percentage", ascending = False)

    #Take all the features that have a non-null number of missing values
    percent_na = percent_na[percent_na["Percentage"]!=0]

    # Take all the corresponding descriptions in meta_data
    descriptions = [x for name in percent_na.index 
                    for x in meta_data.loc[meta_data["Feature name"] == name, "Feature description"]]

    #Gives a description for elevation_threshold which is not present in meta_data
    if data.elevation_th.isna().sum() >0 : 
        descriptions.insert(0,"Elevation threshold") 

    percent_na["Description"] = descriptions

    return percent_na


# Show histograms above of boxplots according to a list of features, a transformation and potential a filter

def show_hist(data, features, figsize=(20,80), n_cols = 5, function=None, log = False, mask_func = None):
    n_rows = 2 * np.ceil(len(features) / n_cols).astype(int)
    
    fig, axes = plt.subplots(nrows = n_rows, 
                             ncols = n_cols, 
                             figsize = figsize,
                             gridspec_kw={"height_ratios": int(n_rows / 2) * [0.5, 0.3]})

    for i, feat in enumerate(features):
        row = i // n_cols
        col = i % n_cols
    
        hist_ax = axes[2 * row, col]
        box_ax = axes[2 * row + 1, col]
        
        data_plot = data[feat]
        
        if mask_func:
            mask = mask_func(data_plot)
            data_plot = data_plot[mask]
            
            
        if function:
            data_plot = function(data_plot)
    
        hist_ax.hist(data_plot, bins = 50, edgecolor = "black", log = log)
        hist_ax.set_title(feat, fontsize=10, fontweight = "bold")
        sns.despine(ax=hist_ax)

        sns.boxplot(x=data_plot, orient="h", ax=box_ax, fliersize=2)
        box_ax.set(yticks=[])
        sns.despine(ax=box_ax, left=True)
    
    plt.tight_layout()
    plt.show()
    
    
# Loads the data used at all models training part

def load_data():
    
    data_tr = pd.read_csv("../datasets/train_data.csv")
    data_te = pd.read_csv("../datasets/test_data.csv")

    X_tr = data_tr.drop("dangerLevel", axis = 1).values
    y_tr = data_tr["dangerLevel"].values

    X_te = data_te.drop("dangerLevel", axis = 1).values
    y_te = data_te["dangerLevel"].values
    
    return X_tr, X_te, y_tr, y_te 



# Shows a table of the results of a GridSearchCV

def show_results(grid_search, grids):
    
    dictionnary = dict()
    
    for keys in list(grids[0].keys()):
        dictionnary[str(keys)] = grid_search.cv_results_["param_" + str(keys)]
        
    dictionnary["mean validation score"] = grid_search.cv_results_["mean_test_score"]
    dictionnary["std validation score"] = grid_search.cv_results_["std_test_score"]
    dictionnary["mean train score"] = grid_search.cv_results_["mean_train_score"]
    dictionnary["std train score"] = grid_search.cv_results_["std_train_score"]
    
    results = pd.DataFrame(dictionnary)
    
    return results


# Shows the validation and training curves of a GridSearchCV

def show_curves(results, grids, parameter, yticks = np.arange(0,1,0.1), mask = None, plot = True):
    
    if mask is not None:
        results = results[mask]
        
        
    mean_tr = results["mean train score"]
    mean_valid = results["mean validation score"]
    std_tr = results["std train score"]
    std_valid = results["std validation score"]
    
    if plot is not True:
        plt.semilogx(results[parameter], mean_tr, label="training accuracy")
        plt.semilogx(results[parameter], mean_valid, label="validation accuracy")
        
    else: 
        plt.plot(results[parameter], mean_tr, label="training accuracy")
        plt.plot(results[parameter], mean_valid, label="validation accuracy")

    best_param = results.loc[mean_valid.idxmax(), parameter]
    plt.scatter(best_param, mean_valid.max(), marker="x", c="red", zorder=10)
    
    length = grids[0]

    plt.fill_between(grids[0][parameter], mean_tr - std_tr, mean_tr + std_tr, alpha=0.2)
    plt.fill_between(grids[0][parameter], mean_valid - std_valid, mean_valid + std_valid, alpha=0.2)
    plt.title("Best {}: {} with {:.1f}% accuracy".format(parameter, best_param, 100 * mean_valid[mean_valid.idxmax()]))
    plt.yticks(yticks)
    plt.xticks(grids[0][parameter])
    plt.ylabel("accuracy")
    plt.xlabel(parameter)
    plt.legend()
    plt.show()
    


# Displays a confusion matrix

def confusion_matrix(y_test, y_pred, model):
    
    fig, ax = plt.subplots(1,1, figsize = (6,4))
    
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels = ["1","2","3","4"], normalize = "true",
        xticks_rotation = 45, ax=ax)
    
    ax.set_title("Confusion Matrix for the {} model".format(model), 
              fontsize = 12, 
              fontweight = "bold")
    plt.show()
        
        
        
# Displays a classification report
    
def class_report(y_test, y_pred):
    
    report = classification_report(y_true = y_test, 
                                   y_pred = y_pred,
                                   labels = [1, 2, 3, 4],
                                   zero_division = 0.0, 
                                   target_names = ["1","2","3","4"])
    print(report)
    

