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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

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
'''
def normalize_data(df, particle_types):
    import pandas as pd
    type_df = pd.DataFrame()
    for x in particle_types:
        df1 = df[df.type == x]
        type_df = pd.concat([type_df, df1])
    return type_df
'''
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
        mask = df["type"].isin(buckets[key])
        df.loc[mask,'type'] = key
        # for every loc in DF where it is true, input what is defined
        # rows, columns: take type column and change to defined
        #EXAMPLE:
        # buckets = {"protein" : ["protein", 'dense globular', 'dense fibral', 'translucent ring-like', 'dense ring-like', 'translucent fibral', 'translucent globular'], "silicon oil": ['silicone oil', 'multi si oil','silicone oil agg.']}
        # rebucket_data(df, buckets=buckets)
        # bucket_types = df.type.unique()
        # print(bucket_types)

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




def confusion(rfc, X, Y, types):
    """
    Display
    Parameters
    ----------
    rfc = random forest model that is used for predictions
    # remove: title_options = list of two grouped strings, (confusion matrix name, normalized or not)
    X = df, Testing X data
    Y = df, Testing Y data
    types = list, all unique types of particles in df
    types = df['type'].unique()

    Returns: non-normalized and normalized Confusion matrices and heat maps of model predictions
    -------

    """
    title_options = [("Confusion Matrix", None), ("Normalized Confusion Matrix", "true")]
    for title, normalize in title_options:
        disp = ConfusionMatrixDisplay.from_estimator(rfc, X, Y, display_labels=types, include_values=False,
                                                     cmap=plt.cm.Blues, normalize=normalize,
                                                     xticks_rotation= 'vertical')
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        plt.tight_layout()
    plt.tight_layout()
    plt.show()

def feature_analysis(rfc, X):
    """


    """
    feature_importance = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=True)
    # use full feature model to evaluate importance of features and rank them low to high

    print('Feature importances: ', rfc.feature_importances_)
    # values of relative feature importance scores, useful for threshold determination

    print(sns.barplot(x=feature_importance, y=feature_importance.index))  # visualization of feature importance
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title("Visualizing Important Features", fontsize=15, pad=15)
    plt.tight_layout()
    plt.show()

def random(df, feat_to_drop, filename):
    rcf = RandomForestClassifier(random_state=0)
    print('Parameters currently in use:\n')
    pprint(rcf.get_params())

    # used grid search cv instead, compare
    # can ask it for the best model not just best parameters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(5, 1000, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    pprint(random_grid)


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rcf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rcf_random = RandomizedSearchCV(estimator=rcf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                    random_state=0, n_jobs=-1)
    # Fit the random search model

    X = df.drop(feat_to_drop, axis=1)
    Y = df['type'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    rcf_random.fit(X_train, Y_train)
    rcf_random.get_params()
    print('best parameters =', rcf_random.best_params_)
    best_random = rcf_random.best_estimator_
    with open(filename, "wb") as f:
        pickle.dump(best_random, f)


def grid(df, feat_to_drop, filename):

    # Create the parameter grid based on the results of random search
    param_grid = {
        'bootstrap': [False],
        'max_depth': [200, 400, 600],
        'max_features': [2, 3],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [4, 6],
        'n_estimators': [500, 1000, 1500, 2000]
    }

    # Create a base model
    rfc_grid = RandomForestClassifier()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rfc_grid, param_grid=param_grid,
                               cv=5, n_jobs=-1, verbose=2)
    print('grid search =', grid_search)

    X = df.drop(feat_to_drop, axis=1)
    Y = df['type'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    grid_search.fit(X_train, Y_train)
    grid_search.best_params_
    print('best parameters for grid search =', grid_search.best_params_)

    best_grid = grid_search.best_estimator_
    prediction = best_grid.predict(X_test)
    accuracy_score(Y_test, prediction)
    best_grid_accuracy = accuracy_score(Y_test, prediction)
    print('best grid accuracy score = ', accuracy_score(Y_test, prediction))

    rfc_grid_base = RandomForestClassifier(criterion='gini', max_depth=3, random_state=0)
    rfc_grid_base.fit(X_train, Y_train)
    prediction = rfc_grid_base.predict(X_test)
    base_grid_accuracy = rfc_grid_base.score(Y_test, prediction)

    # print('best parameters =',rcf_random.best_params_)

    print('best parameters for grid search =', grid_search.best_params_)


    print('base grid accuracy score =', base_grid_accuracy)

    print('Improvement of {:0.2f}%'.format(100 * (best_grid_accuracy - base_grid_accuracy) / base_grid_accuracy))
    print('Improvement of {:0.2f}% from gridsearch against randomsearchcv'.format(
        100 * (best_grid_accuracy - base_accuracy) / base_accuracy))
    with open(filename, "wb") as f:
        pickle.dump(best_grid, f)

