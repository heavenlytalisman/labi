from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


cats = ['comp.graphics', 'rec.sport.baseball', 'sci.med', 'talk.politics.misc']
news = fetch_20newsgroups(subset='all', categories=cats,
remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target,
test_size=0.2, random_state=42)


vec = CountVectorizer(stop_words='english')
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)


model = MultinomialNB().fit(X_train_vec, y_train)
print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vec)))


sample = ["The patient needs medicine and treatment."]
pred = model.predict(vec.transform(sample))
print("Predicted Category:", news.target_names[pred[0]])
