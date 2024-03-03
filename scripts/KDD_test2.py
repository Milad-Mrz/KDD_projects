import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Visualizing usage of bike per month.
plt.figure(figsize=(15, 5))
sns.boxplot(x='mnth', y='cnt', data=df, palette='Set2')
plt.title('Bike Rental Count per Month')
plt.xlabel('Month')
plt.ylabel('Count')
plt.show()


# Visualizing usage of bike per hour.
plt.figure(figsize=(25, 5))
sns.boxplot(x='hr', y='cnt', data=df, palette='Set2')
plt.title('Hourly Bike Rental Count')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.show()


# Visualizing impact of temperature on bike rental.
plt.figure(figsize=(25, 5))
sns.boxplot(x='temp', y='cnt', data=df, palette='Set2')
plt.title('Temperature vs Bike Rental')
plt.xlabel('Temperature')
plt.ylabel('Count')
plt.show()


# Visualizing windspeed vs rental count of bike.
plt.figure(figsize=(10, 5))
plt.scatter(df['windspeed'], df['cnt'], color='orange')
plt.title('Windspeed vs Rental Counts')
plt.xlabel('Windspeed')
plt.ylabel('Rental Counts')
plt.show()


# Visualizing Weather situation vs bike rentals.
plt.figure(figsize=(15, 5))
sns.boxplot(x='weathersit', y='cnt', data=df, palette='Set2')
plt.title('Weather vs Bike Rental')
plt.xlabel('Weather')
plt.xticks(np.arange(3), ('1: Clear', '2: Mist + Cloudy', '3: Light Rain/Snow'))
plt.ylabel('Count')
plt.show()


# Visualizing season vs bike rentals.
plt.figure(figsize=(10, 5))
sns.boxplot(x='season', y='cnt', data=df, palette='Set2')
plt.title('Season vs Rental Counts')
plt.xlabel('Season')
plt.xticks(np.arange(4), ('1: Winter', '2: Spring', '3: Summer', '4: Fall'))
plt.ylabel('Rental Counts')
plt.show()


