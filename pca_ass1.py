#########################################################################

#                    ASSIGNMENT (PCA) I:-
#                    ====================


#########################################################################



"""Perform hierarchical and K-means clustering on the dataset.
 After that, perform PCA on the dataset and extract the first 3 principal
 components and make a new dataset with these 3 principal components
 as the columns. Now, on this new dataset, perform hierarchical and 
 K-means clustering. Compare the results of clustering on the original
 dataset and clustering on the principal components dataset
 (use the scree plot technique to obtain the optimum number of 
  clusters in K-means clustering and check if you’re getting similar
  results with and without PCA)."""
 
'''
Business Objectives:
    
    Business Objective:

    maximize: A composite measure of wine quality, potentially derived from 
              expert ratings or sensory evaluations.

    minimize: Production cost

    constrains: Consistency of Brand Image
'''

'''data dictionary'''
#type-show the which type are wine
#alcohol-it show the alocohol % on wine
#malic-it show the malic % on wine
#Ash- show present or not in wine ,it inorgainic content
#Alcalinity- show the alclonity % on wine
#Magnesium - show the Magnesium in wine
#Phenols - Phenols in the wine in % 
#Flavanoids -flavanoids content in the red wine in mg
#Nonflavanoids - Nonflavanoids content in the wine
#Proanthocyanins -Proanthocyanins key metabolites that explain wine sensorial character (bitterness and astringency) and red wine color changes during aging
#Color - it show wine color
#Hue -it s show Hue of wine
#Dilution -Dilution of the wine
#Proline-The amount of proline in the wine can vary from O to about 90 % of the total nitrogen


# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reading the wine dataset
alco = pd.read_csv("wine.csv")

# Exploratory data analysis
print("Dataset Shape:", alco.shape)
print("Columns:", alco.columns)
alco.info()
print("Descriptive Statistics:\n", alco.describe())

# Handling outliers using Winsorization
columns_to_winsorize = ['Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Proanthocyanins', 'Color', 'Hue']
for column in columns_to_winsorize:
    winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.4, variables=[column])
    df1 = winsor.fit_transform(alco[[column]]).astype(int)
    alco[column] = df1
columns_to_winsorize
df1


# Applying masking for outliers in 'Proanthocyanins' column
IQR = alco.Proanthocyanins.quantile(0.75) - alco.Proanthocyanins.quantile(0.25)
lower_limit = alco.Proanthocyanins.quantile(0.25) - 1.5 * IQR
upper_limit = alco.Proanthocyanins.quantile(0.75) + 1.5 * IQR
outlier = np.where(alco.Proanthocyanins > upper_limit, upper_limit,
                   np.where(alco.Proanthocyanins < lower_limit, lower_limit, alco.Proanthocyanins))
alco.Proanthocyanins = outlier
outlier

# Visualizing boxplots after outlier removal
sns.boxplot(alco)

# Converting ordinal values into dummy variables
df1 = pd.get_dummies(alco)

# Applying normalization to the dataset
def normalize_data(i):
    x = (i - i.min()) / (i.max() - i.min())
    return x

df_norm = normalize_data(df1)
df_norm.drop(['Ash', 'Proanthocyanins'], axis=1, inplace=True)

####################################################################
#let apply the clustering algoritham 
#-------------------------------------
from sklearn.cluster import KMeans
TWSS=[]
k=list(range(2,8))
k
#we donts know the k value os the cluster that way 
#we aplying all the possible number in the range of 2 to 8 in TWWS
#and plot the fig and understand the k value
for i in k:
    kmean=KMeans(n_clusters=i).fit(df_norm)
    TWSS.append(kmean.inertia_)
TWSS   
plt.plot(k,TWSS,'-ro')
plt.plot(k, TWSS,'ro-')
plt.xlabel("no of cluster")
plt.ylabel("total within ss")

#from the fig we understant the cluster value k=3
model=KMeans(n_clusters=3).fit(df_norm).labels_
mb=pd.Series(model)
#add one columns in original dataset and add new column of cluster
alco['clust']=mb

# Applying KMeans clustering algorithm to determine the optimal number of clusters
TWSS = []
k = list(range(2, 8))
for i in k:
    kmean = KMeans(n_clusters=i).fit(df_norm)
    TWSS.append(kmean.inertia_)

# Plotting the Total Within Sum of Squares (TWSS) to identify the optimal cluster count
plt.plot(k, TWSS, '-ro')
plt.plot(k, TWSS, 'ro-')
plt.xlabel("Number of Clusters")
plt.ylabel("Total Within Sum of Squares")

# From the plot, it is observed that the optimal number of clusters is 3
# Applying KMeans clustering with k=3 and adding the cluster information to the original dataset
model = KMeans(n_clusters=3).fit(df_norm).labels_
alco['clust'] = pd.Series(model)

# Applying Principal Component Analysis (PCA) to reduce feature dimensions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_norm)

pca = PCA(n_components=6)
pca_values = pca.fit_transform(X_scaled)
pca_values
#Output
array([[-3.64723729, -1.22043595,  0.01898743,  0.76329298,  0.25570108,
        -0.8499216 ],
       [-3.07865531,  0.10754147, -1.254332  , -0.15635253, -1.0953982 ,
        -0.57815524],
       [-2.47675083, -0.36129757,  0.1369775 , -0.51892384,  0.3296485 ,
        -0.38246983],
       ...,
       [ 2.79549297, -3.22118021,  0.30129742,  0.80201993, -0.09489902,
        -0.53488797],
       [ 2.51469486, -2.54460144, -0.51777328,  1.14107888,  0.75274821,
         0.11505276],
       [ 3.39170868, -2.4886564 ,  0.56252121, -1.2449983 ,  0.79685133,
         0.24425528]])


# Creating a DataFrame for the PCA components
pca_data = pd.DataFrame(pca_values, columns=['comp0', 'comp1', 'comp2', 'comp3', 'comp4', 'comp5'])
pca_data
#Output
 comp0     comp1     comp2     comp3     comp4     comp5
0   -3.647237 -1.220436  0.018987  0.763293  0.255701 -0.849922
1   -3.078655  0.107541 -1.254332 -0.156353 -1.095398 -0.578155
2   -2.476751 -0.361298  0.136978 -0.518924  0.329648 -0.382470
3   -3.867339 -2.425341  1.107730 -0.417598 -0.276804  0.825823
4   -1.493445 -0.101805  0.708280  0.429516  1.347963 -0.840049
..        ...       ...       ...       ...       ...       ...
173  3.265104 -2.136260  0.211888 -1.295673 -0.347755 -1.132484
174  2.820821 -1.715933  0.256742  0.118857  0.238621  0.170304
175  2.795493 -3.221180  0.301297  0.802020 -0.094899 -0.534888
176  2.514695 -2.544601 -0.517773  1.141079  0.752748  0.115053
177  3.391709 -2.488656  0.562521 -1.244998  0.796851  0.244255













