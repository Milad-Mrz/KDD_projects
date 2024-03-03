#functions related to Data pre-processing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss


from sklearn.preprocessing import MinMaxScaler

def plot_Rents_vs_Season_Attributes(data, file_name):
    #instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
    # setting colors
    colors = ['steelblue', 'springgreen', 'gold', 'orangered'] 
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    attr_list = [ 'hr', 'temp', 'hum', 'windspeed']


    fig1, axs = plt.subplots(2, 2, figsize=(12, 8))
    sub = 'Number of Rents per Season vs Attributes'+ str(file_name)
    fig1.suptitle(sub, fontsize=16)

    # Iterate over each subplot
    for i, ax in enumerate(axs.flat):
        attr_name = attr_list[i]
        # grouping number of rent by season and attributes: 
        cnt_attr = data.groupby(['season', attr_name])['cnt'].mean(numeric_only=True)
        # Iterate over each season and plot the line
        for j, season in enumerate(seasons):
            color = colors[j]
            cnt_season_attr = cnt_attr.xs(j+1, level='season')
            cnt_season_attr.plot(ax=ax, color=color, label=season)

        ax.set_xlabel(attr_name)
        ax.set_ylabel('Number of Rents')
        ax.legend()
    file_name = '../images/cnt_per_Season_vs_Attributes'+ file_name +'.png'
    fig1.savefig(file_name, dpi=300, bbox_inches='tight')

def balance_data_oversampler(data):
    # Extract the features and target variable
    X = data.drop(['cnt'], axis=1)
    y = data['cnt']

    # Instantiate the RandomOverSampler
    oversampler = RandomOverSampler()

    # Perform oversampling on the data
    X_balanced, y_balanced = oversampler.fit_resample(X, y)

    # Create a new balanced data frame
    df_balanced = pd.concat([X_balanced, y_balanced], axis=1)

    return df_balanced



def balance_data_undersampler(data, bins=4):
    # Create bins for the 'cnt' variable
    data['cnt_bins'] = pd.cut(data['cnt'], bins=bins, labels=False)

    ###
    import matplotlib.pyplot as plt

    # Plot histogram of the cnt_bins column
    plt.hist(data['cnt'], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel('Bins')
    plt.ylabel('Number of Instances')
    plt.title('Histogram of Binned DataFrame')
    plt.grid(True)
    plt.show()
    ###

    X = data.drop(['cnt', 'cnt_bins'], axis=1)
    y = data['cnt_bins']

    # Compute class counts for each bin
    class_counts = y.value_counts()

    # Calculate the minimum number of instances among all bins
    min_instances = class_counts.min()
    print(min_instances)

    # Create a sampling strategy dictionary with the desired number of instances per bin
    sampling_strategy = {bin_label: min_instances for bin_label in class_counts.index}

    # Instantiate the RandomUnderSampler
    undersampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)

    # Perform undersampling on the data
    X_balanced, y_balanced = undersampler.fit_resample(X, y)

    # Create a new balanced data frame
    df_balanced = pd.concat([X_balanced, y_balanced], axis=1)

    # Plot histogram of the cnt_bins column
    plt.hist(df_balanced['cnt'], bins=20, color='#ef233c', edgecolor='black')
    plt.xlabel('Bins')
    plt.ylabel('Number of Instances')
    plt.title('Histogram of Binned DataFrame')
    plt.grid(True)
    plt.show()
    ###

    # Drop the temporary 'cnt_bins' column
    df_balanced.drop('cnt_bins', axis=1, inplace=True)

    return df_balanced


def balance_data_nearmiss(data, bins=10):

    data = data.drop(['dteday'], axis=1)
    # Create bins for the 'cnt' variable

    data['cnt_bins'] = pd.cut(data['cnt'], bins=bins, labels=False)

    X = data.drop(['cnt', 'cnt_bins'], axis=1)
    y = data['cnt_bins']

    # Compute class counts for each bin
    class_counts = y.value_counts()

    # Calculate target number of samples per bin
    target_samples_per_bin = np.mean(class_counts)

    # Compute the sampling strategy as a dictionary
    sampling_strategy = {bin_label: int(target_samples_per_bin) for bin_label in class_counts.index}

    # Instantiate the NearMiss sampler
    sampler = NearMiss(sampling_strategy=sampling_strategy, version=1)

    # Perform undersampling on the data
    X_balanced, y_balanced = sampler.fit_resample(X, y)

    # Create a new balanced data frame
    df_balanced = pd.concat([X_balanced, y_balanced], axis=1)

    # Drop the temporary 'cnt_bins' column
    df_balanced.drop('cnt_bins', axis=1, inplace=True)

    return df_balanced


def normalize_dataframe(data, exlcude_list):
    #instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
    # Get a list of non-text columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in exlcude_list]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize the non-text columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    data[numeric_cols] = data[numeric_cols].round(5)

    data['dteday'] = data.index / data.index.max()
    data['dteday'] = data['dteday'].round(7)

    return data

def save_dataframe(data, file_name):
    #instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
    # Get a list of non-text columns



    file_name = '../data/output/hour_'+ file_name 
    numeric_cols = ['temp','hum','windspeed', 'cnt']
    data[numeric_cols] = data[numeric_cols].round(3)

    with open(file_name+'.txt', 'w') as file:
        # Perform your loop
        for column in numeric_cols :
            content = str(column) + ', ' +  str(data[column].min()) + ', ' + str(data[column].max()) + '\n'
            file.write(content)

    data['instant'] = data.index
    data.to_csv(file_name+'.csv', index=False)

    return data