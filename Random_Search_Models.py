import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
from pprint import pprint
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from utilsF import random

feat_to_drop = ['type','step','experiment','particle_id', 'ml_type_proba']

df = pd.read_csv('categorized_particles.csv')
df2 = pd.read_csv('fiber.csv')
df3 = pd.read_csv('air_bubble_and_fiber.csv')
df4 = pd.read_csv('categorized_particles_101.csv')

types=df4['type'].unique()
print(types)

def group_particles(df4, particle_type):
    type_df=df.loc[df["type"]==particle_type]
    print("number of {} particles = {}".format(particle_type, type_df.shape[0]))
    return type_df
for x in types:
    group_particles(df,x)

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
rebucket_data(df4, buckets=buckets)
bucket_types = df4.type.unique()
print(bucket_types)

random(df4,feat_to_drop,"human_best_model.pkl")