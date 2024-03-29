import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
import pickle
from pprint import pprint
from utilsF import random, grid

#takes too long to run, too many particles?

feat_to_drop = ['type','step','experiment','particle_id']

df = pd.read_csv('Particle Data/categorized_particles.csv')


types=df['type'].unique()
print(types)

def group_particles(df, particle_type):
    type_df=df.loc[df["type"]==particle_type]
    print("number of {} particles = {}".format(particle_type, type_df.shape[0]))
    return type_df
for x in types:
    group_particles(df, x)


def rebucket_data(df, buckets=None):
    """
   Converts an ML data dataframe into a new one where the type column is simplified according to the buckets dict.
        Parameters
    ----------
    df : dataframe
        Input data to transform.

    buckets : dict
    import IPython; IPython.embed()
    """
    for key in buckets.keys():
        #print(buckets[keys])
        mask = df["type"].isin(buckets[key])
        df.loc[mask,'type'] = key
        # for every loc in DF where it is true, input what is defined
        # rows, columns: take type column and change to defined

buckets = {"protein" : ["protein", 'dense globular', 'dense fibral', 'translucent ring-like', 'dense ring-like', 'translucent fibral', 'translucent globular'], "silicon oil": ['silicone oil', 'multi si oil','silicone oil agg.']}
rebucket_data(df, buckets=buckets)
bucket_types = df.type.unique()
#print(bucket_types)

random(df,feat_to_drop,"main_best_model_random.pkl")

grid(df, feat_to_drop, "main_best_model_grid.pkl")