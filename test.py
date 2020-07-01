from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from tnt import TNT

dataset = fetch_20newsgroups(subset="all",
                             shuffle=True,
                             random_state=42)

# vectorizer can accept all scikit's vectorizers.
# 'count' and 'tfidf' are recognized by default
# Note: tnt only works with cosine similarity on l2-normalized vectors!
tnt = TNT(vectorizer="count")  # add kwargs to change default parameters

# the index is built in 13s
tnt.fit(dataset.data)  # feed in a list of strings
# get your search results in 1ms
pprint(tnt.search("computer science", k=5))
