from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 1. Load Specific 4 Categories
cats = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.misc']
news = fetch_20newsgroups(subset='all', categories=cats,
remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target,
test_size=0.2, random_state=42)

# 2. Vectorize text (Word Counts)
vec = CountVectorizer(stop_words='english')
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# 3. Train & Evaluate
model = MultinomialNB().fit(X_train_vec, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vec)))

# 4. Predict New Sample
sample = ["The patient needs medicine and treatment."]
pred = model.predict(vec.transform(sample))
print("Predicted Category:", news.target_names[pred[0]])
