import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


with open('task1.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()


titles = []
genres = []
plot_summaries = []

for line in lines:
    parts = line.strip().split(":::")
    if len(parts) == 4: 
        title, year, genre, plot_summary = parts
        titles.append(title.strip())
        genres.append(genre.strip())
        plot_summaries.append(plot_summary.strip())



x_train, x_test, y_train, y_test = train_test_split(plot_summaries, genres, test_size=0.2, random_state=42)



tfidf_vectorizer = TfidfVectorizer(max_features=5000) 
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)


clf = MultinomialNB()
clf.fit(x_train_tfidf, y_train)


y_pred = clf.predict(x_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#report = classification_report(y_test, y_pred)
#print("Classification Report:\n", report)

#Predict genre for a new plot summary
'''new_plot_summary = ["A group of friends embark on an epic adventure in a fantasy world."]
new_plot_summary_tfidf = tfidf_vectorizer.transform(new_plot_summary)
predicted_genre = clf.predict(new_plot_summary_tfidf)
print(f"Predicted Genre: {predicted_genre[0]}")
'''