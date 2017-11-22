import pandas as pd
# import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

df = pd.read_csv('/home/thuwarakesh/projects/NBSentiment/review_data.csv')


def clean(text):
    stops = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    stop_free = [t for t in text.lower().split() if t not in stops]
    punc_free = [t for t in stop_free if t not in exclude]
    normalized = [lemma.lemmatize(t) for t in punc_free]

    return normalized


normalized_reviews = df.review.map(clean)

word_dummies = pd.get_dummies(normalized_reviews.apply(pd.Series).stack()).sum(level=0)

word_counts = word_dummies.sum().sort_values(ascending=False)

# fig, ax = plt.subplots()
# fig.set_figheight(10)
# word_counts[:50].plot(kind='barh')

print(word_counts[:50])
