import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from nltk.stem import WordNetLemmatizer  
import re
nltk.download('wordnet')

df = pd.read_csv('D:/Spam/spam.csv', encoding=('ISO-8859-1'), low_memory=False)
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df = df[['v1', 'v2']]
df.columns = ['target', 'text']

lemmatizer = WordNetLemmatizer()  

def preprocess_text(text):
    
    text = text.lower()
    text = ' '.join([word for word in text.split() if word.isalpha()])
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

df['text'] = df['text'].apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) 
X = vectorizer.fit_transform(df['text'])
y = df['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_resampled, y_resampled)


y_pred = rf_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
