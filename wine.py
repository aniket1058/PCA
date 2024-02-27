#                       ASSIGNMENT WINE.CSV
#                       ===================

########################################################################
########################################################################

'''
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

''
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('wine.csv')
df

#######################################
df.columns
'''
Index(['Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
       'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
       'Dilution', 'Proline'],
      dtype='object')
'''
#######################################
df.dtypes
'''
Type                 int64
Alcohol            float64
Malic              float64
Ash                float64
Alcalinity         float64
Magnesium            int64
Phenols            float64
Flavanoids         float64
Nonflavanoids      float64
Proanthocyanins    float64
Color              float64
Hue                float64
Dilution           float64
Proline              int64
dtype: object

All the  datatype is of numeric type
'''

#######################################
a=df.describe()
# there scale diffrence in mean and std and the mean and median are  near approx same
# but show some variaion as the standard deviation is showing diffrence with mean
# so the datapoints are scatter from median

#####################################
# Check for the null values

n=df.isnull()
n.sum()
'''
Out[39]: 
Type               0
Alcohol            0
Malic              0
Ash                0
Alcalinity         0
Magnesium          0
Phenols            0
Flavanoids         0
Nonflavanoids      0
Proanthocyanins    0
Color              0
Hue                0
Dilution           0
Proline            0
dtype: int64

The dataframe doesn't contain any null value'
'''
#####################################
# Visualize the Data
# To undestand the corelaion between the datapoints and the columns we plot
# some plots

import seaborn as  sns

sns.pairplot(df)
# from the pairplot observe that the data is more scatter and the relation between
# the columns are quite similar

# To identify if there is any outlier in columns we plot the boxplot
sns.boxplot(df)
# There is and outlier present in the magnesium column

# to analyze whether the columns follow patern or not we draw the heatmap 

corr=df.corr()
sns.heatmap(corr)
# from the heamap i can understand that the diagonal colour of are same so the 
# columns follow some pattern 

#####################################
# So there is outlier is present and also the the column shows skewness propery
# and there is scale difference in mean and std so we use standardization technique as we are going to use 
# PCA

# Standardization
# initialize the scalar
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
df=scalar.fit_transform(a)
dataset=pd.DataFrame(df)
res=dataset.describe()
# in the resvariable we will see that the mean value is almost value 
#Standard deviation is zero

#########################

# Model Building

#For visualzing the cluster of  the above dataframe we  have to draw
# Dendodron first then we cluster the datapoints

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# linkage function give the hierarchical and Agglomotive clustering
 

z=linkage(dataset,method='complete',metric='euclidean')

plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering')
plt.xlabel('Index')
plt.ylabel('Disance')
#sch is help to draw 
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#appying agglomerative clustering choose 1 as a cluster from dendogram

# In dedrogram is not show the clustering it only shows how many clusters are there

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=2,linkage='complete',affinity='euclidean').fit(dataset)

#apply labels to the cluster
h_complete.labels_
# so these all are in the form of array we have to convert the Series
cluster_labels=pd.Series(h_complete.labels_)
# so these all are in the form of array we have to convert the Series
cluster_labels=pd.Series(h_complete.labels_)

df['clust']=cluster_labels
df

##########################################
# K-Means Clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df=pd.read_csv('wine.csv')
df
TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(dataset)
    
    TWSS.append(kmeans.inertia_)
    
    '''
    kmeans inertia also known as sum odf sqares methos
    .It measures all the datapoints from the centroid of the point.
    it differentiate between observed value and predicted value
    '''
    
TWSS
# Plot a elbow curve
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusers')
plt.ylabel('Total within SS')

model=KMeans(n_clusters=3)
model.fit(dataset)
model.labels_
mb=pd.Series(model.labels_)
type(mb)
df['clust']=mb
df.head()
d=df.iloc[:,[5,0,1,2,3,4]]
d
#################### PCA #####################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#Normalize the numeric data
uni_normal=scale(df)
uni_normal

pca=PCA(n_components=3)
pca_values=pca.fit_transform(uni_normal)

#The amount of variance that each PCA explain

var=pca.explained_variance_ratio_
var

#Commulative Variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1
#Variance plot for PCA component obtained
plt.plot(var1,color='red')
#PCA Scores
pca_values

pca_data=pd.DataFrame(pca_values)
pca_data.columns='comp0','comp1','comp2','comp3','comp4','comp5'

final=pd.concat([df.clust,pca_data],axis=1)

#Visualize the dataframe
ax=final.plot(x='comp0',y='comp1',kind='scatter',figsize=(12,8))
final[['comp0','comp1','clust']].apply(lambda x:ax.text(*x),axis=1)

####################################################################