Text Neighbor Tracker
=====================

A fast and easy-to-build search engine for your text datasets.

```bash
pip install tnt-learn
```

TNT works by vectorizing your corpus using Scikit's popular Tf-Idf or Count vectorizers. An approximate k-nearest neighbor search is made when querying, using an algorithm inspired by the cluster pruning algorithm (see reference at the end of this page). Multiple indexes can be merged into one instance (named kTNT), to make searches more accurate, albeit slower.

The algorithm works with datasets as large as millions of documents, without consuming much RAM. Its speed is some order of magnitudes faster than a KDTree, both in indexing and querying time. It is tailored for sparse and highly dimensional data such as text. It is not meant to replace a proper search engine: its purpose is to save you (a lot of) time and costs, when you only wish to explore your datasets or provide a temporary micro-service. It does not currently support insertion after index construction.

Example usage
=============

```python
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from tnt import TNT

dataset = fetch_20newsgroups(subset="all",
                             shuffle=True,
                             random_state=42)

# vectorizer can accept all scikit's vectorizers.
# 'count' (default) and 'tfidf' are recognized keywords.
# Note: tnt only works with cosine similarity on l2-normalized vectors
tnt = TNT(vectorizer="count", verbose=True)

# index is built in 13 seconds
tnt.fit(dataset.data)  # feed in a list of strings
# get your search results in 1 ms
pprint(tnt.search("computer science", k=5))
```

You can avoid returning texts and distances, as list lookups are quite slow in Python:

```python
tnt.search("computer science", k=5, return_text=False, return_distance=False)
```

CountVectorizer is the fastest vectorizer method, and works well for short texts such as tweets or emails. You can also use TfidfVectorizer or other sklearn-like algorithms. All additional keyword parameters are automatically passed to the vectorizer:

```python
tnt = TNT(vectorizer="tfidf", ngram_range=(0, 1), stop_words={"french"})
```

You can use your own vectorizer as well:

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    binary=False,
    max_features=100000,
    stop_words={"english", "french"},
    min_df=4, max_df=.5)
tnt = TNT(vectorizer=vectorizer)
```
Keep in mind that the current nearest neighbor search only supports sparse matrices and cosine similarity. It also requires your document vectors to be l2-normalized. The reason is that the current implementation gets some of its speed-up by considering the (cosine) similarity as a simple dot product between two vectors.

Persistence
-----------

You can save the TNT instance using the appropriate method:
```python
tnt.save("tnt.p")
```
And load:
```python
from tnt import load
tnt = load("tnt.p")
tnt.search("BLM")
```

Grow a forest of TNTs
---------------------

If accuracy is paramount, you can build a kTNT instance, which can be thought of as a forest of TNTs: using the same vectorizer, multiple indexes are built and queried. This is of course a trade-off between accuracy and speed.

```python
from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from tnt import kTNT, load

dataset = fetch_20newsgroups(subset="all",
                             shuffle=True,
                             random_state=42)
tnt = kTNT(k=5)  # you can also add vectorizer=...
tnt.fit(dataset.data)  # index building is 5 times slower
tnt.search("I like speed", k=100))  # but search is only a tiny bit slower
tnt.save("ktnt.p")

tnt = load("ktnt.p")
tnt.search("here we go again!")
```

kTNT is the recommended way to use TNT when dealing with very short documents such as tweets, as picking sqrt(N) documents is sometimes not enough to capture short texts consisting of uncommon words. Using k=2 or 3 can ensure that all semantically reasonable queries get some match with its "leaders" (see details on the cluster pruning algorithm at the end of this page to understand this sentence).

depth
-----

All TNT and kTNT instances have a depth (>1) parameter, controlling the depth of the search tree. depth=2 is enough for most datasets, but for massive datasets, the parameter can go up to 3, 4 or more (not recommended).

```python
tnt = kTNT(k=5, depth=3)  # 2 by default
tnt.fit(...)
```

Requirements
============
Apart from Dill, Numpy, Scipy and obviously Scikit-learn, TNT also makes use of my own Nearset package (https://github.com/kerighan/nearset). Nearsets are efficient ordered sets of key-value pairs, sorted at insertion using a custom comparator function, such as the distance of a vector to a given query.
You can install it using:

```bash
pip install nearset -U
```

Further information
===================

Information related to the cluster pruning neighbor search:

https://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html
https://dl.acm.org/doi/10.1145/1265530.1265545
