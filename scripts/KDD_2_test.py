import pandas as pd
from KDD_f2 import *

df_under = pd.read_csv("./data/output/hour_S_bal_under_norm.csv")
df_orginal = pd.read_csv("./data/output/hour_original.csv")
df_over = pd.read_csv("./data/output/hour_S_bal_over_norm.csv")

#df_over = df_over.sample(n=30000, random_state=42)
#final_models, model_list = modelSelector(df_over, 'over30k', 'cnt')

#final_models, model_list = modelSelector(df_under, 'under', 'cnt')
#final_models, model_list = modelSelector(df_orginal, 'orgin', 'cnt')

#df_over10k = df_over.sample(n=10000, random_state=42)
#final_models, model_list = modelSelector(df_over10k, 'over10k', 'cnt')

#df_over20k = df_over.sample(n=18000, random_state=42)
#final_models, model_list = modelSelector(df_over20k, 'over', 'cnt')

#df_over30k = df_over.sample(n=30000, random_state=42)
#final_models, model_list = modelSelector(df_over30k, 'over30k', 'cnt')

#df_over40k = df_over.sample(n=40000, random_state=42)
#final_models, model_list = modelSelector(df_over40k, 'over40k', 'cnt')

#/home/milad/Desktop/KDD/KDD_biking/data/output/hour_bal_over.csv

"""df_over = pd.read_csv("./data/output/hour_bal_over.csv")
df_over10k = df_over.sample(n=10000, random_state=42)
final_models, model_list = modelSelector(df_over10k, 'over10k_B', 'cnt')

df_over20k = df_over.sample(n=20000, random_state=42)
final_models, model_list = modelSelector(df_over20k, 'over20k_B', 'cnt')

df_over30k = df_over.sample(n=30000, random_state=42)
final_models, model_list = modelSelector(df_over30k, 'over30k_B', 'cnt')

df_over = pd.read_csv("./data/output/hour_S.csv")
df_over10k = df_over.sample(n=10000, random_state=42)
final_models, model_list = modelSelector(df_over10k, 'over10k_S', 'cnt')

df_over20k = df_over.sample(n=20000, random_state=42)

df_over = pd.read_csv("./data/output/hour_S.csv")
final_models, model_list = modelSelector(df_over, 'over20k_S', 'cnt')
"""

df_over40k = df_over.sample(n=40000, random_state=42)
final_models, model_list = modelSelector(df_over40k, 'over40k_B', 'cnt')


