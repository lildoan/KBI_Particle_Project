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
from utilsF import random, confusion, feature_analysis, group_particles, rebucket_data, \
    make_histogram, make_scatter
from Random_Forest_function import random_forest_model

# file to run best model on bucketed human categorized data

# df = pd.read_csv('categorized_particles.csv')
# df2 = pd.read_csv('fiber.csv')
# df3 = pd.read_csv('air_bubble_and_fiber.csv')
df4 = pd.read_csv('Particle Data/categorized_particles_101.csv')

types = df4['type'].unique()
print(types)

for x in types:
    group_particles(df4, x)

buckets = {"protein": ["protein", 'dense globular', 'dense fibral', 'translucent ring-like',
                       'dense ring-like', 'translucent fibral', 'translucent globular'],
           "silicon oil": ['silicone oil', 'multi si oil', 'silicone oil agg.']}

rebucket_data(df4, buckets=buckets)
bucket_types = df4.type.unique()
print(bucket_types)

fig1 = plt.figure(figsize=(8, 6))
n_bins = 10
ax1 = fig1.add_subplot(1, 1, 1)
parameter_list = list(df4.columns)

def make_histogram(fig, df, particle_type, parameter):
    type_df = df.loc[df["type"] == particle_type]
    ax1.set_title("{} particle distribution".format(parameter))
    ax1.set_xlabel(parameter)
    ax1.set_ylabel('number of particles')
    ax1.hist(type_df[parameter], n_bins, label=particle_type)
    legend = plt.legend(markerscale=10)

for x in bucket_types:
    make_histogram(fig1, df4, x, "circularity")

def make_scatter(fig, df, particle_type, parameter_1, parameter_2):
    type_df = df.loc[df["type"] == particle_type]
    ax1.scatter(type_df[parameter_1], type_df[parameter_2], s=0.1, label=particle_type)
    ax1.set_xlabel(parameter_1)
    ax1.set_ylabel(parameter_2)
    legend = plt.legend(markerscale=10)

fig2 = plt.figure(figsize=(8, 6))
ax1 = fig2.add_subplot(1, 1, 1)
for x in bucket_types:
    make_scatter(fig2, df4, x, "circularity", "intensity_mean")

#random_forest_model(df4, ['type', 'step', 'experiment', 'particle_id', 'ml_type_proba'], 0.3, 0.07)



best_random = pickle.load(open("human_best_model_random.pkl", "rb"))

feat_to_drop = ['type', 'step', 'experiment', 'particle_id', 'ml_type_proba'] #, 'edge_particle', 'x_right', 'x_left',
                 #'y_bottom', 'time_stamp_s', 'frame_num', 'y_top', 'particle_num']
#'edge_particle', 'x_right', 'x_left', 'y_bottom', 'time_stamp_s', 'frame_num', 'y_top', 'particle_num'
X = df4.drop(feat_to_drop, axis=1)
Y = df4['type'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Best Model
prediction = best_random.predict(X_test)
best_random_accuracy = best_random.score(X_test, Y_test)
print('best accuracy score =', best_random_accuracy)
print('train random =', best_random.score(X_train, Y_train))
# f1_score(Y_test, prediction)

# Base Model
rfc_base = RandomForestClassifier(criterion='gini', random_state=0)
rfc_base.fit(X_train, Y_train)
prediction_base = rfc_base.predict(X_test)
base_accuracy = rfc_base.score(X_test,Y_test)

print('base accuracy score =', base_accuracy)
print('train base =', rfc_base.score(X_train,Y_train))
print('Improvement of {:0.2f}%'.format(100 * (best_random_accuracy - base_accuracy) / base_accuracy))

confusion(best_random, X_test, Y_test, bucket_types)

feature_analysis(best_random,X)

plt.show()
