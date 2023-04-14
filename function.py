#this is a test
#hi this is amanda teehee i have a crush on lilly <3

#from Random_Forest_function import random_forest_model
import pandas as pd
from utilsF import normalize_data


df_h = pd.read_csv('categorized_particles_101.csv') #data categorized by humans, model trained on this
df_h.type.values.reshape(-1, 1).shape

df = pd.read_csv('categorized_particles.csv') #data categorized by other ML model, new data to test model on
df.type.values.reshape(-1, 1).shape

protein_data = normalize_data(df, ['protein', 'dense globular', 'dense fibral', 'translucent ring-like', 'dense ring-like', 'translucent fibral', 'translucent globular'])
protein_data.shape
silicone_data = normalize_data(df, ['silicone_oil', 'multi_si_oil', 'silicone_oil_agg'])
schlieren_data = normalize_data(df, ['schlieren_lines'])
other_data = normalize_data(df, ['air_bubble', 'glass', 'air_aggregate'])

df1 = pd.concat([protein_data,silicone_data,schlieren_data,other_data])
print(df1.shape)

print(df1.head())
