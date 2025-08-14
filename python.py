import pandas as pd

#data collection
df_test = pd.read_csv(r'C:\Users\pulin\OneDrive\Documents\Desktop\data science workspace\true.csv')
df_train = pd.read_csv(r'C:\Users\pulin\OneDrive\Documents\Desktop\data science workspace\fake.csv')

#data cleaning and preprocessing
#merging data sets
df = pd.concat([df_test, df_train])
df.reset_index(drop=True, inplace=True)
#explore the dataset
print(df.head())
print(df.info())
print(df.describe())
#missing values
print(df.isnull().sum())
print(df.columns)
#splitting the data
from sklearn.model_selection import train_test_split
X = df['text']
Y = df['subject']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)

#exploratory data analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print(df.describe())
#data vizualisation
#sns.set()
#plt.figure (figsize=(10,6))
#sns.histplot(df['text'])
#plt.title('distribution of column name')
#plt.show()
#plt.savefig('histogram.png')
#correlation heatmap
#plt.figure(figsize=(10,8))
#sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
#plt.title('correlation heatmap')
#plt.show()
#plt.savefig('heatmap.png')
#box plot
#plt.figure(figsize=(10,6))
#sns.boxplot(df['text'])
#plt.title('Box plot of column name')
#plt.show()
#plt.savefig('boxplot.png')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

texts = [
    "Scientists Discover Cure for Common Cold"
    "Local Politician Embroiled in Scandal"
    "New Study Shows Coffee Causes Cancer"
    "Government Announces New Economic Stimulus Packages"
]

labels = ["fake", "real", "fake", "real"]
print(len(texts))
print(len(labels))

#if length are different
if len(texts) != len(labels):
    min_len = min(len(texts), len(labels))
    texts = texts[:min_len]
    labels = labels[:min_len]
#texts are new articles and labels are real or fake
from sklearn.model_selection import train_test_split

if len(texts) < 2:
    print("not enough samples to split. useingentire dataset for training")
    x_train, y_train = texts, labels
    x_test, y_test = [], []
else:
    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
#model fitting
model = DecisionTreeClassifier()
model.fit(x_train_vec, y_train)
preds = model.predict(x_test_vec)
print("Accuracy:", accuracy_score(y_test, preds))
print(len(texts))
print(len(labels))

import pickle 
with open('model.pkl','wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl','wb') as f:
    pickle.dump(vectorizer, f)
    

