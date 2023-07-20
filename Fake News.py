import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Downloading the NLTK stopwords for English language
import nltk
nltk.download('stopwords')

# Printing the stopwords in English
print(stopwords.words('english'))

# Loading the dataset into a pandas DataFrame
news_dataset = pd.read_csv('train.csv')

# Counting the number of missing values in the dataset
news_dataset.isnull().sum()

# Replacing the null values with an empty string
news_dataset = news_dataset.fillna('')

# Merging the author name and news title into a new 'content' column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Separating the data and label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Initializing the PorterStemmer for stemming words
port_stem = PorterStemmer()

# Function to perform stemming on the content
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Applying stemming on the 'content' column
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Separating the data and label after stemming
X = news_dataset['content'].values
Y = news_dataset['label'].values

# Converting the textual data to numerical data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Creating a Logistic Regression model
model = LogisticRegression()

# Training the model on the training data
model.fit(X_train, Y_train)

# Calculating accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data: ', training_data_accuracy)

# Calculating accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data: ', test_data_accuracy)

# Selecting the first data point from the test data for model testing
X_new = X_test[0]

# Making predictions using the model on the new data point
prediction = model.predict(X_new)
print(prediction)

# Printing the result of the model testing
if prediction[0] == 0:
    print('The news is Real')
else:
    print('The news is Fake')

# Printing the actual label of the first data point in the test data
print(Y_test[0])
