import pandas as pd
from matplotlib import pyplot as plt


def group_particles(df, particle_type):
    type_df=df.loc[df["type"]==particle_type]
    print("number of {} particles = {}".format(particle_type, type_df.shape[0]))
    return type_df

def make_scatter(fig, df, particle_type, parameter_1, parameter_2):
    type_df=df.loc[df["type"]==particle_type]
    ax1.scatter(type_df[parameter_1], type_df[parameter_2], s=0.1, label = particle_type)
    ax1.set_xlabel(parameter_1)
    ax1.set_ylabel(parameter_2)
    legend=plt.legend(markerscale = 10)

def make_histogram(fig, df, particle_type, parameter):
    type_df=df.loc[df["type"]==particle_type]
    ax1.set_title("{} particle distribution".format(parameter)) 
    ax1.set_xlabel(parameter)
    ax1.set_ylabel('number of particles')
    ax1.hist(type_df[parameter], n_bins, label = particle_type)
    legend=plt.legend(markerscale = 10)

# Create buckets to group proteins 
def normalize_data(df, particle_types):
    import pandas as pd
    type_df = pd.DataFrame()
    for x in particle_types:
        df1 = df[df.type == x]
        type_df = pd.concat([type_df, df1])
    return type_df

'''
protein_data = normalize_data(df, ['protein', 'dense globular', 'dense fibral', 'translucent ring-like', 'dense ring-like', 'translucent fibral', 'translucent globular'])
protein_data.shape
silicone_data = normalize_data(df, ['silicone_oil', 'multi_si_oil', 'silicone_oil_agg'])
schlieren_data = normalize_data(df, ['schlieren_lines'])
other_data = normalize_data(df, ['air_bubble', 'glass', 'air_aggregate'])
'''

##DEBUGGING CODE
# if __name__ == "__main__":
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     df = pd.read_csv('categorized_particles.csv')
#     fig = plt.figure(figsize=(20,30))
#     types = df['type'].unique()
#     for x in types:
#         make_scatter(fig, df, "glass", "intensity_max", "circularity")
        
        
#def make_scatter(fig, df, particle_type, parameter_1, parameter_2):
#     type_df=df.loc[df["type"]==particle_type]
#    # for x in parameter_list: 
#     for i in range(0,5):
#         for n in range (0,2):
#             import pdb; pdb.set_trace()
#             axi=fig.add_subplot(3,2,n+1)
#             axi.set_xlabel(parameter_1)
#             axi.set_ylabel(parameter_2)
#             axi.scatter(type_df[parameter_1], type_df[parameter_2], s=0.1, label="particle_type")
# def make_scatter(fig, df, particle_type, parameter_1, parameter_2):
#     ax1.scatter(particle_type['parameter 1'], particle_type['parameter 2'], s=0.1, label="particle_type")

    # return the particle type group with the size of the group
# fig=plt.figure(figsize=(20,40))
# ax1=fig.add_subplot(1,1,1)
# for every particle type
    # make a plot against "parameter1", "parameter 2", onto defined figure, particle type
# data locate: 


 # fig=plt.figure(figsize=(20,40))
    #ax1=fig.add_subplot(1,1,1)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def confusion(rfc, title_options, X, Y, types):
    """

    Parameters
    ----------
    rfc = random forest model that is used for predictions
    title_options = list of two grouped strings, (confusion matrix name, normalized or not)
    X = df, Testing X data
    Y = df, Testing Y data
    types = list, all unique types of particles in df

    Returns: non-normalized and normalized Confusion matrices and heat maps of model predictions
    -------

    """
    for title, normalize in title_options:
        disp = ConfusionMatrixDisplay.from_estimator(rfc, X, Y, display_labels=types, include_values=False,
                                                     cmap=plt.cm.plasma, normalize=normalize,
                                                     xticks_rotation='vertical')
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)




