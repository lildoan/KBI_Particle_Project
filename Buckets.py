
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

fig1 = plt.figure(figsize=(8,6))
n_bins = 10
ax1 = fig1.add_subplot(1,1,1)
parameter_list = list(df4.columns)
def make_histogram(fig, df, particle_type, parameter):
    type_df=df.loc[df["type"]==particle_type]
    ax1.set_title("{} particle distribution".format(parameter))
    ax1.set_xlabel(parameter)
    ax1.set_ylabel('number of particles')
    ax1.hist(type_df[parameter], n_bins, label = particle_type)
    legend=plt.legend(markerscale = 10)

for x in bucket_types:
    make_histogram(fig1, df4, x, "circularity")
plt.show()
def make_scatter(fig, df, particle_type, parameter_1, parameter_2):
    type_df=df.loc[df["type"]==particle_type]
    ax1.scatter(type_df[parameter_1], type_df[parameter_2], s=0.1, label = particle_type)
    ax1.set_xlabel(parameter_1)
    ax1.set_ylabel(parameter_2)
    legend=plt.legend(markerscale = 10)
fig2 = plt.figure(figsize=(8,6))
ax1 = fig2.add_subplot(1,1,1)
for x in bucket_types:
    make_scatter(fig2, df4, x, "circularity", "intensity_mean")
plt.show()
