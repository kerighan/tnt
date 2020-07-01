Text Neighbor Tracker
=====================

A fast and easy-to-build search engine for your textual datasets.

```bash
pip install tnt-learn
```

TNT works by vectorizing your corpus using Scikit's popular Tf-Idf or Count vectorizers. An approximate k-nearest neighbor search is done when querying, using an algorithm inspired by Stanford's cluster pruning algorithm. Multiple indexes can be merged in one TNT (named kTNT) instance, to make searches more accurate (and slower).

The algorithm works with datasets as large as millions of documents. Its speed is some order of magnitudes faster, both in indexing and querying time than a KDTree. It is tailored for sparse and highly dimensional data such as text. It is not meant to replace a proper search engine. Its purpose is to save you (a lot of) time and costs, when you only wish to explore your datasets or provide a micro-service.

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
# Note: tnt only works with cosine similarity on l2-normalized vectors!
tnt = TNT(vectorizer="count", verbose=True)

# the index is built in 13 seconds
tnt.fit(dataset.data)  # feed in a list of strings
# get your search results in 1 ms
pprint(tnt.search("computer science", k=5))  # this returns text as well as index
```

You can avoid returning texts and distances, as list lookups are quite slow in Python:

```python
tnt.search("computer science", k=5, return_text=False, return_distance=False)
```

CountVectorizer is the fastest method, and works well for short texts. You can also use TfidfVectorizer. Additional parameters are automatically passed to the TfidfVectorizer.

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
Keep in mind that the nearest neighbor search only support sparse matrices, and only works with cosine similarity. It also requires your document vectors to be l2-normalized. The reason is that its implementation gets some of its speed-up by computing the cosine similarity as a simple dot product between two vectors.

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

A forest of TNTs
----------------

If accuracy is paramount, you can build a kTNT instance, which is like a forest of TNTs: using the same vectorizer, multiple indexes are built and queried. This is of course a trade-off between accuracy and speed.

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

kTNT is the recommended way to use TNT, as picking sqrt(N) documents is sometimes not enough to capture enough cases for uncommon words.

depth
-----

All TNT and kTNT instances have a depth (>1) parameter, controlling the depth of the search tree. depth=2 is enough for most datasets. For massive datasets, the parameter can go up to 3, 4 or more (not recommended).

```python
tnt = kTNT(k=5, depth=3)  # 2 by default
tnt.fit(...)
```

Requirements
============
Apart from Dill, Numpy, Scipy and obviously Scikit-learn, TNT also makes use of my Nearset package (https://github.com/kerighan/nearset). Nearsets are ordered sets of key-value pairs, sorted at insertion using a custom comparator function.
You can install it using:

```bash
pip install nearset -U
```

Further information
===================

Information related to the cluster pruning neighbor search:
https://nlp.stanford.edu/IR-book/html/htmledition/cluster-pruning-1.html
https://dl.acm.org/doi/10.1145/1265530.1265545
