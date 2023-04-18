from Random_Forest_function import random_forest_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utilsF import normalize_data

"""
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

"""

from timeit import default_timer as timer
start = timer()

#-------------------------------------------------------------------------------------------------------------

df_h = pd.read_csv('categorized_particles_101.csv') #data categorized by humans, model trained on this
df_h.type.values.reshape(-1, 1).shape

feat_to_drop = ['type','step','experiment','particle_id','ml_type_proba'] #features dropped from human categorized, has extra ml_type_proba feature

#-------------------------------------------------------------------------------------------------------------

df = pd.read_csv('categorized_particles.csv') #data categorized by other ML model, new data to test model on
df.type.values.reshape(-1, 1).shape

test_feat_drop = ['type','step','experiment','particle_id'] #features dropped from ML categorized

#-------------------------------------------------------------------------------------------------------------

random_forest_model(df_h, feat_to_drop, 0.3, 0.1)

end = timer()
print("run time =",end - start," seconds")

'''''
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('categorized_particles.csv')
df.type.values.reshape(-1, 1).shape
feat_to_drop = ['type','step','experiment','particle_id']
test_feat_drop = ['type','step','experiment','particle_id']
max_depth_range = list(range(1,9))
accuracy = []
X=df.drop(feat_to_drop, axis=1)
    Y=df['type'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
for depth in max_depth_range:
    rcf = RandomForestClassifier(max_depth = depth, random_state = 0)
    rcf.fit(X_train, Y_train)
    score = rcf.score(X_test, Y_test)
    accuracy.append(score)
    #print(accuracy)
    plt.plot(max_depth_range, accuracy)
'''''''
