#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
my_csv = Path("C:/Users/adika/OneDrive/Desktop/ub/summer23/DIC_587/project/shooting-1982-2023.csv")
df = pd.read_csv(my_csv.resolve(), sep=',')


# In[2]:


df['location'] = df['location'].astype('string')
df['summary'] = df['summary'].astype('string')
df['location.1'] = df['location.1'].astype('string')
df['where_obtained'] = df['where_obtained'].astype('string')
df['weapon_type']=df['weapon_type'].astype('string')
df['weapon_details']=df['weapon_details'].astype('string')
df['race']=df['race'].astype('string')
df['gender']=df['gender'].astype('string')
df['type']=df['type'].astype('string')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].astype('string')


# In[3]:


df[['Month','Day','year']] = df['date'].str.split('/', expand=True)
df[['City','State']] = df['location'].str.split(',', expand=True)


# In[4]:


df['gender'] = df['gender'].replace('M', 'Male')
df['gender'] = df['gender'].replace('F', 'Female')


# In[5]:


df['location.1'] = df['location.1'].replace('Other\n', 'other')
df['location.1'] = df['location.1'].replace('religious', 'Religious')
df['location.1'] = df['location.1'].replace('\nwork', 'work')
df['location.1'] = df['location.1'].replace('Other', 'other')


# In[6]:


df['race'] = df['race'].replace('white', 'White')
df['race'] = df['race'].replace('Black','black')
df['race'] = df['race'].replace('unclear','Other')
df['race'] = df['race'].replace('White ', 'White')


# In[7]:


df['prior_signs_mental_health_issues'] = df['prior_signs_mental_health_issues'].replace('', 'unknown')
df['prior_signs_mental_health_issues'] = df['prior_signs_mental_health_issues'].replace('Yes', 'yes')
df['prior_signs_mental_health_issues'] = df['prior_signs_mental_health_issues'].replace('Unclear', 'unknown')
df['prior_signs_mental_health_issues'] = df['prior_signs_mental_health_issues'].replace('TBD', 'unknown')
df['prior_signs_mental_health_issues'] = df['prior_signs_mental_health_issues'].replace('Unknown', 'unknown')


# In[8]:


df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('Unknown', 'unknown')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('Yes', 'yes')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('TBD', 'unknown')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('Kelley passed federal criminal background checks; the US Air Force failed to provide information on his criminal history to the FBI', 'unknown')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('\nYes', 'yes')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('Yes ("some of the weapons were purchased legally and some of them may not have been")', 'yes')
df['weapons_obtained_legally'] = df['weapons_obtained_legally'].replace('Yes ', 'yes')


# In[9]:


def divide_ages_into_groups(df, column_name):
    # Define the age ranges and corresponding labels
    age_ranges = [
        (0, 17, 'Child'),
        (18, 25, 'Young Adult'),
        (26, 40, 'Adult'),
        (41, 60, 'Middle-aged'),
        (61, float('inf'), 'Senior')
    ]

    # Create a new column to store the age groups
    df['Age Group'] = ''

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        age = row[column_name]

        # Check the age against each age range
        for range_start, range_end, group_label in age_ranges:
            if range_start <= age <= range_end:
                df.at[index, 'Age Group'] = group_label
                break

    return df
df = divide_ages_into_groups(df, 'age_of_shooter')


# In[10]:


df.shape


# In[11]:


df.head(3)


# In[12]:


df.dtypes


# In[13]:


df.drop(['case','type','weapon_details','location','date','type'], axis=1)


# In[14]:


df["freq_count"] = df.groupby("State")["State"].transform('count')
df.head(2)


# In[15]:


import plotly.express as px
import json
us_states = json.load(open("C:/Users/adika/Downloads/us-states.json",'r')) 
us_states["features"][1]["properties"]


# In[16]:


df1=df
df1=df1.dropna(subset=["State"])
df1["State"].isnull().count()


# In[17]:


df["State"].unique()


# In[18]:


df1=df
df1["freq_count"] = df.groupby("State")["State"].transform('count')


# In[19]:


df1['freq_count'].unique()


# In[20]:


df1["State"] = df1["State"].replace(" Tennessee", "Tennessee")
df1['State'] = df1["State"].replace("tennessee", "Tennessee")
df1['State'] = df1["State"].replace(" California", "California")
df1['State'] = df1["State"].replace(" Michigan", "Michigan")
df1['State'] = df1["State"].replace(" Virginia", "Virginia")
df1['State'] = df1["State"].replace(" Colorado", "Colorado")
df1['State'] = df1["State"].replace(" North Carolina", "North Carolina")
df1['State'] = df1["State"].replace(" Indiana", "Indiana")
df1['State'] = df1["State"].replace(" Illinois", "Illinois")
df1['State'] = df1["State"].replace(" Alabama", "Alabama")
df1['State'] = df1["State"].replace(" Maryland", "Maryland")
df1['State'] = df1["State"].replace(" Oklahoma", "Oklahoma")
df1['State'] = df1["State"].replace(" Texas", "Texas")
df1['State'] = df1["State"].replace(" New York", "New York")
df1['State'] = df1["State"].replace(" Georgia", "Georgia")
df1['State'] = df1["State"].replace(" Missouri", "Missouri")
df1['State'] = df1["State"].replace(" Wisconsin", "Wisconsin")
df1['State'] = df1["State"].replace(" New Jersey", "New Jersey")
df1['State'] = df1["State"].replace(" Florida", "Florida")
df1['State'] = df1["State"].replace(" Ohio", "Ohio")
df1['State'] = df1["State"].replace(" Pennsylvania", "Pennsylvania")
df1['State'] = df1["State"].replace(" Nevada", "Nevada")
df1['State'] = df1["State"].replace(" Washington", "Washington")
df1['State'] = df1["State"].replace(" Lousiana", "Lousiana")
df1['State'] = df1["State"].replace(" Kansas", "Kansas")
df1['State'] = df1["State"].replace(" Oregon", "Oregon")
df1['State'] = df1["State"].replace(" South Carolina", "South Carolina")
df1['State'] = df1["State"].replace(" D.C.", "D.C.")
df1['State'] = df1["State"].replace(" Connecticut", "Connecticut")
df1['State'] = df1["State"].replace(" Minnesota", "Minnesota")
df1['State'] = df1["State"].replace(" Arizona", "Arizona")
df1['State'] = df1["State"].replace(" Kentucky", "Kentucky")
df1['State'] = df1["State"].replace(" Nebraska", "Nebraska")
df1['State'] = df1["State"].replace(" Utah", "Utah")
df1['State'] = df1["State"].replace(" Mississippi", "Mississippi")
df1['State'] = df1["State"].replace(" Massachusetts", "Massachusetts")
df1['State'] = df1["State"].replace(" Hawaii", "Hawaii")
df1['State'] = df1["State"].replace(" Arkansas", "Arkansas")
df1['State'] = df1["State"].replace(" Iowa", "Iowa") 


# In[21]:


df1 = df.dropna(subset=["State"])
state_id_map_dict = {}
for feature in us_states["features"]:
    state_id_map_dict[feature["properties"]["name"]] = feature["id"]
df1["id"] = df1["State"].map(state_id_map_dict)
df1["State"].value_counts()


# In[22]:


state_id_map_dict


# In[23]:


df1["freq_count"] = df1["freq_count"].astype(int) 


# In[24]:


df1["freq_count"]


# In[25]:


df['State']


# In[54]:


import plotly.express as px
import json
us_states = json.load(open("C:/Users/adika/Downloads/us-states.json",'r'))
fig = px.choropleth(df1, locations="id", scope="usa",hover_name="State", geojson=us_states,color = "freq_count", 
                    color_continuous_scale="teal")
fig.show()


# In[27]:


from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color='black', stopwords=stopwords, max_words=500, max_font_size=40, random_state=42).generate(str(df['summary']))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[28]:


df.dtypes


# In[29]:


wordcloud = WordCloud(background_color='black',stopwords=stopwords,max_words=300,max_font_size=30,random_state=45).generate(str(df['mental_health_details']))
plt.figure(figsize=(12,8))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[30]:


df['where_obtained'] = df['where_obtained'].astype('string')
df['mental_health_details'] = df['mental_health_details'].astype('string')
df['weapon_type'] = df['weapon_type'].astype('string')
df['weapon_details']=df['weapon_details'].astype('string')


# In[31]:


df['weapon_type'].unique()


# In[32]:


df['weapon_type'] = df['weapon_type'].replace('semiautomatic handgun\n', 'semiautomatic handgun')
df['weapon_type'] = df['weapon_type'].replace('multiple\n', 'multiple')
df['weapon_type'] = df['weapon_type'].replace('Handgun','handgun')
df['weapon_type'] = df['weapon_type'].replace('handgun\n','handgun')
df['weapon_type'] = df['weapon_type'].replace('semi-automatic handgun', 'semiautomatic handgun')
df['weapon_type'] = df['weapon_type'].replace('One semiautomatic handgun', 'semiautomatic handgun')
df['weapon_type'] = df['weapon_type'].replace('Shotgun', 'shotgun')
df['weapon_type'] = df['weapon_type'].replace('One rifle (assault)', 'Assault rifle')
df['weapon_type'] = df['weapon_type'].replace('One rifle', 'rifle')
df['weapon_type'] = df['weapon_type'].replace('One revolver', 'revolver')
df['weapon_type'] = df['weapon_type'].replace('One shotgun', 'shotgun')
df['weapon_type'] = df['weapon_type'].replace('semiautomatic assault weapon (Details pending)', 'assault weapon')
df['weapon_type'] = df['weapon_type'].replace('Rifle', 'rifle')


# In[33]:


df['weapon_type'].unique()


# In[34]:


df['weapon_type'].value_counts()
plt.figure(figsize=(10,8))
df['weapon_type'].value_counts().plot.pie(explode=None, autopct = '%1.1f%%', shadow=False)
plt.title('weapon types')


# In[35]:


df.plot.box(column=["age_of_shooter"], figsize=(8, 8))


# In[36]:


age_data = df["age_of_shooter"]
plt.figure(figsize=(10, 6))
age_data.value_counts().sort_index().plot(kind='bar', color='black')
plt.title("Age of Shooters")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


# In[37]:


age_data = df['age_of_shooter']
plt.hist(age_data, bins=20, edgecolor='black')
plt.title('Age of Shooters')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[38]:


print("Maximum number of fatalities in a mass shooting : ", np.max(df['fatalities']))
print("Minimum number of fatalities in a mass shooting : ", np.min(df['fatalities']))
print("Average number of fatalities in any mass shooting : ", int(np.mean(df['fatalities'])))
fat_count = df.fatalities
plt.figure(figsize=(10,5))
plt.scatter(range(len(fat_count)), np.sort(fat_count.values), alpha=0.7)
plt.title("Fatalities count in mass shooting")
plt.xlabel("Index")
plt.ylabel("Count")
plt.show()


# In[39]:


print("Maximum number of injured in a mass shooting : ", np.max(df['injured']))
print("Minimum number of injured in a mass shooting : ", np.min(df['injured']))
print("Average number of injured in any mass shooting : ", int(np.mean(df['injured'])))
inj_count = df.injured
plt.figure(figsize=(10,5))
plt.scatter(range(len(inj_count)), np.sort(inj_count.values), alpha=0.7)
plt.title("Injuries count in mass shooting")
plt.xlabel("Index")
plt.ylabel("Count")
plt.show()


# In[40]:


fatalities_per_year = df.groupby("year")["fatalities"].sum()
victims_per_year = df.groupby("year")["total_victims"].sum()
yr=df['year']


# In[41]:


year_data = df['year']
injured_data = df["injured"]
fatal = df['fatalities']
vic = df['total_victims']
plt.plot(year_data, injured_data, marker='o', linestyle='-', color='blue')
plt.title('Number of Injured People per Year')
plt.xlabel('Year')
plt.ylabel('Number of Injured People')
plt.xticks(rotation=90)
plt.show()


# In[79]:


fatal = df['fatalities']
plt.plot(year_data, fatal, marker='o', linestyle='-', color='red')
plt.title('Number of Fatalities per Year')
plt.xlabel('Year')
plt.ylabel('Number of Fatalities')
plt.xticks(rotation=90)
plt.show()


# In[80]:


vic = df['total_victims']
plt.plot(year_data, vic, marker='o', linestyle='-', color='green')
plt.title('Number of total victims per Year')
plt.xlabel('Year')
plt.ylabel('Number of total victims')
plt.xticks(rotation=90)
plt.show()


# In[44]:


from sklearn.linear_model import LinearRegression
X = df[['year']] 
y = df['fatalities'] 
model = LinearRegression()
model.fit(X.values, y)
future_years = [[2024], [2025], [2026], [2027],[2028],[2029],[2030],[2031],[2032],[2033]]  
future_deaths = model.predict(future_years)
for year, deaths in zip(future_years, future_deaths):
    print(f"Year: {year[0]}, Predicted Deaths: {deaths}")


# In[45]:


plt.plot(future_years, future_deaths, marker='o', linestyle='-', color='red')
plt.title('Predicted Number of Deaths in Future Years')
plt.xlabel('Year')
plt.ylabel('Number of Deaths')
plt.show()


# In[68]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X = df["year"].values.reshape(-1, 1)
y = df["fatalities"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[71]:


import matplotlib.pyplot as plt
X_train_flattened = X_train.flatten()
X_test_flattened = X_test.flatten()
plt.scatter(X_train_flattened, y_train, c='blue', label='Training Data')
plt.scatter(X_test_flattened, y_test, c='green', label='Testing Data')
plt.scatter(X_test_flattened, y_pred, c='red', label='Predicted Data')
plt.xlabel('Year')
plt.ylabel('Fatalities')
plt.title('SVM Classification for Mass Shooting Data')
plt.xticks(rotation=90)
plt.legend()
plt.show()


# In[84]:


from sklearn.neighbors import KNeighborsRegressor
selected_columns = ['year', 'fatalities']
dfk = df[selected_columns]
train_dfk, test_dfk = train_test_split(dfk, test_size=0.2, random_state=42)
train_X = train_dfk[['year']]
train_y = train_dfk['fatalities']
test_X = test_dfk[['year']]
test_y = test_dfk['fatalities']
knn = KNeighborsRegressor(n_neighbors=16)
knn.fit(train_X, train_y)
predictions = knn.predict(test_X)

test_X_flattened = test_X.values.flatten()

plt.scatter(test_X_flattened, test_y, color='blue', label='Actual')
plt.scatter(test_X_flattened, predictions, color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Fatalities')
plt.title('Actual vs Predicted Fatalities in Mass Shootings')
plt.xticks(rotation = 90)
plt.legend()
plt.show()


# In[59]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
dfd = df[['fatalities']]
scaler = StandardScaler()
dfd_scaled = scaler.fit_transform(dfd)
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(dfd_scaled)
plt.scatter(dfd['fatalities'], [0] * len(dfd), c=clusters)
plt.xlabel('Fatalities')
plt.show()


# In[60]:


from sklearn.ensemble import IsolationForest
selected_columns = ['fatalities', 'injured']  
selected_data = df[selected_columns]
selected_data = selected_data.dropna()
model = IsolationForest(contamination=0.01)  
model.fit(selected_data)
anomaly_predictions = model.predict(selected_data)
selected_data['anomaly'] = anomaly_predictions
anomalies = selected_data[selected_data['anomaly'] == -1]
print(anomalies)


# In[61]:


plt.scatter(selected_data['injured'], selected_data['fatalities'],  label='Data Points')
anomalies = selected_data[selected_data['anomaly'] == -1]
plt.scatter(anomalies['injured'], anomalies['fatalities'], color='red', marker='x', label='Anomalies')
plt.xlabel('Injured')
plt.ylabel('Fatalities')
plt.title('Anomaly Detection - Mass Shooting Data')
plt.show()


# In[ ]:




