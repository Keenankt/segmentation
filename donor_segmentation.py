### import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import gender_guesser.detector as gdg
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.patches as mpatches

### read donor data
df = pd.read_csv('donor_data.csv')

##################### median property value as basis for net worth/capacity to donate
### create dictionary to map postcode to suburb name
vic_postcodes = []
vic_postcodes = pd.read_csv('australian_postcodes.csv')
vic_postcodes = vic_postcodes[vic_postcodes["state"] == "VIC"]
vic_postcodes = vic_postcodes[vic_postcodes["postcode"] < 4000]
vic_postcodes = vic_postcodes[["postcode","locality"]]
vic_postcodes.dropna(inplace=True)
postcode_dict = vic_postcodes.set_index('locality')['postcode'].to_dict()
### create dictionary to map postcode to median price of suburb
postcode_prices = []
postcode_prices = pd.read_csv("C:/Users/kkuli/repos/donor_segmentation/median_property.csv")
postcode_prices['suburb'] = postcode_prices['suburb'].map(postcode_dict)
price_dict = postcode_prices.set_index('suburb')['med_price'].to_dict()
### add new column for median property value of each donor based on postcode
df['price'] = df['postcode'].replace(to_replace=price_dict)
def placeholder_price(price):
    if price < 10000:
        new_price = 300000
        return new_price
    else:
        return price
df['price'] = df['price'].apply(placeholder_price)


########################## split names and guess gender
split_names = df['name'].str.split(pat= ' ',expand=True)
df[['first','last','last2','last3']] = split_names
df.drop({'name','last2','last3'},axis=1,inplace=True)
## create gender detector function
detector = gdg.Detector() 
def guess_gender(first):
    gender = detector.get_gender(first)
    return gender
## apply detector to frame
df['gender'] = df['first'].apply(guess_gender)


######################### exploration start
plt.style.use('seaborn-v0_8-bright')
## distributions
fig, ax = plt.subplots(1,3)
fig.suptitle("Distribution of Individual Donor Characteristics") 
sns.boxplot(y=df['annual_donation'],ax=ax[0],color='tab:blue')
ax[0].set_ylabel("Annual Donation Total $")
ax[0].title.set_text('How much they donate')
sns.boxplot(y=df['num_donation'],ax=ax[1],color='tab:cyan')
ax[1].set_ylabel("How frequently they donate")
ax[1].title.set_text('Donation Frequency')
sns.boxplot(y=df['price'],ax=ax[2],color='tab:pink')
ax[2].set_ylabel("Median Property Price of Suburb $ ,000,000")
ax[2].title.set_text('Estimated capacity for giving')
plt.show()
## correlations
fig, ax = plt.subplots(1,1)
sns.scatterplot(x=df['price'],y=df['annual_donation'],ax=ax,color='tab:blue')
fig.suptitle("Total Annual Donations vs Estimated capacity for giving")
ax.set_xlabel("Median Property Price of Suburb $ ,000,000")
ax.set_ylabel("Annual Donation Total $")
a, b = np.polyfit(df['price'],df['annual_donation'],1)
plt.plot(df['price'],(a*df['price']+b),c='tab:green')
plt.show()

##### segmentation start
## normalize data
df = df[['annual_donation','num_donation','price']]
scaler = preprocessing.StandardScaler()
df_norm = scaler.fit_transform(df)
df_norm = pd.DataFrame(df_norm, columns=['annual_donation','num_donation','price'])

## investigate optimal clusters
#distortions = []
#test_range = range(1,10)
#for k in test_range:
    #test_model = KMeans(n_clusters=k).fit(df_norm)
    #distortions.append(test_model.inertia_)

#plt.subplot(1,1,1)
#plt.plot(test_range,distortions)
#plt.title("Kmeans cluster model Intertia")
#plt.show()

## train model
#clusterer = KMeans(n_clusters = 4)
#clusterer_labels = clusterer.fit_predict(df_norm)
custom_map = {0:'tab:blue',1:'tab:orange',2:'tab:green',3:'tab:purple'}
# investigate optimal clusters
dendo_data = linkage(df_norm, method='ward', metric='euclidean')
dendrogram(dendo_data)
plt.title('Visual representation of grouping process')
plt.xlabel('Individual Donors')
plt.ylabel('"Distinctness" of groupings')
plt.xticks(ticks=())
plt.set_cmap('tab10')
plt.show()
# run clustering algorithm and fit data
hierarchical_cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster_labels = hierarchical_cluster.fit_predict(df_norm)
# label raw and normalised data with cluster id
df_norm['cluster'] = cluster_labels
df['cluster'] =  cluster_labels

# plot identified clusters in 3d space
clusterplot = plt.axes(projection='3d')
clusterplot.set_title('Segmented Donor Clusters')
clusterplot.scatter3D(df['price'],df['num_donation'],df['annual_donation'], c=df['cluster'].map(custom_map))
clusterplot.set_xlabel("Median Property Price of Suburb $ ,000,000")
clusterplot.set_ylabel('Number of Donations')
clusterplot.set_zlabel('Annual Donations $')

patch0 = mpatches.Patch(color='tab:blue', label='Segment 0')
patch1 = mpatches.Patch(color='tab:orange', label='Segment 1')
patch2 = mpatches.Patch(color='tab:green', label='Segment 2')
patch3 = mpatches.Patch(color='tab:purple', label='Segment 3')
clusterplot.legend(handles=[patch0,patch1,patch2,patch3])


plt.show()
# distribution of cluster characteristics
plt.suptitle('Donor Characteristics Across Groupings')
plt.subplot(2,2,1)
sns.boxplot(x=df['cluster'],y=df['annual_donation'], palette=custom_map)
plt.title('Annual Donations $')
plt.ylabel('Annual Donations $')
plt.xlabel('')
plt.subplot(2,2,2)
sns.boxplot(x=df['cluster'],y=df['num_donation'], palette=custom_map)
plt.title('Number of Donations')
plt.ylabel('Number of Donations')
plt.xlabel('')
plt.subplot(2,2,3)
sns.boxplot(x=df['cluster'],y=df['price'], palette=custom_map)
plt.title('Estimated Capacity for Giving')
plt.ylabel('Median Property Price of Suburb $ ,000,000')
plt.xlabel('Groups')
plt.subplot(2,2,4)
sns.countplot(x=df['cluster'], palette=custom_map)
plt.title('Number of Donors per Cluster')
plt.xlabel('Groups')
plt.ylabel('Number of Donors')
plt.show()
