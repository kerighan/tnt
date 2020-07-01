from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from .sparnn import Sparnn
from nearset import Nearset
import numpy as np
import time


class TNT(object):
    def __init__(
        self,
        vectorizer="count",
        depth=2,
        normalize=True,
        verbose=True,
        **kwargs
    ):
        if vectorizer == "count":
            self.vectorizer = CountVectorizer(
                ngram_range=kwargs.get("ngram_range", (1, 2)),
                binary=kwargs.get("binary", True),
                stop_words=kwargs.get("stop_words", {"english", "french"}),
                min_df=kwargs.get("min_df", 4),
                **kwargs)
        elif vectorizer == "tfidf":
            normalize = False
            self.vectorizer = TfidfVectorizer(
                ngram_range=kwargs.get("ngram_range", (1, 2)),
                binary=kwargs.get("binary", False),
                stop_words=kwargs.get("stop_words", {"english", "french"}),
                min_df=kwargs.get("min_df", 4),
                **kwargs)
        else:
            self.vectorizer = vectorizer

        self.verbose = verbose
        self.sparnn = Sparnn(depth=depth, verbose=False)
        self.normalize = normalize

    def fit(self, corpus):
        start_time = time.time()
        self.corpus = corpus
        space = self.vectorizer.fit_transform(corpus)
        if self.normalize:
            space = normalize(space)
        self.sparnn.fit(space)

        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"T={elapsed_time:.4f}s")

    def vectorize(self, text):
        vector = self.vectorizer.transform([text])
        if self.normalize:
            vector = normalize(vector)
        return vector

    def search(
        self,
        text,
        k=20,
        threshold=1,
        return_text=True,
        return_distance=True
    ):
        start_time = time.time()

        vector = self.vectorize(text)
        dists, inds = self.sparnn.search(vector, k=k, return_distance=True)
        results = []
        if return_distance and return_text:
            for dist, index in zip(dists, inds):
                if dist >= threshold:
                    break
                results.append({
                    "index": index,
                    "text": self.corpus[index],
                    "distance": dist
                })
        elif return_distance:
            for dist, index in zip(dists, inds):
                if dist >= threshold:
                    break
                results.append({
                    "index": index,
                    "distance": dist
                })
        elif return_text:
            for dist, index in zip(dists, inds):
                if dist >= threshold:
                    break
                results.append({
                    "index": index,
                    "text": self.corpus[index]
                })
        else:
            for dist, index in zip(dists, inds):
                if dist >= threshold:
                    break
                results.append(index)

        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"T={elapsed_time:.4f}s")
        return results

    def save(self, filename):
        import dill
        with open(filename, "wb") as f:
            dill.dump(self, f)


class kTNT():
    def __init__(
        self,
        k=2,
        vectorizer="count",
        depth=2,
        normalize=True,
        verbose=True,
        **kwargs
    ):
        if vectorizer == "count":
            vectorizer = CountVectorizer(
                ngram_range=kwargs.get("ngram_range", (1, 2)),
                binary=kwargs.get("binary", True),
                stop_words=kwargs.get("stop_words", {"english", "french"}),
                min_df=kwargs.get("min_df", 4),
                **kwargs)
        elif vectorizer == "tfidf":
            normalize = False
            vectorizer = TfidfVectorizer(
                ngram_range=kwargs.get("ngram_range", (1, 2)),
                binary=kwargs.get("binary", False),
                stop_words=kwargs.get("stop_words", {"english", "french"}),
                min_df=kwargs.get("min_df", 4),
                **kwargs)
        else:
            vectorizer = vectorizer

        self.verbose = verbose
        self.tnt = []
        self.k = k
        for i in range(k):
            self.tnt.append(
                TNT(depth=depth,
                    vectorizer=vectorizer,
                    normalize=normalize,
                    verbose=False,
                    **kwargs))

    def fit(self, corpus):
        start_time = time.time()
        self.corpus = corpus
        for tnt in self.tnt:
            tnt.fit(self.corpus)

        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"T={elapsed_time:.4f}s")

    def search(
        self,
        text, 
        k=20,
        threshold=1,
        return_text=True,
        return_distance=True
    ):
        start_time = time.time()

        # vectorize once
        vector = self.tnt[0].vectorize(text)

        # nearset is used to sort data by distance
        nearset = Nearset(lambda x: x, max_size=k)
        for i in range(self.k):
            dists, inds = self.tnt[i].sparnn.search(
                vector, k=k,
                return_distance=True)
            for dist, ind in zip(dists, inds):
                nearset[ind] = dist

        results = []
        if return_text and return_distance:
            for index, dist, _ in nearset:
                if dist >= threshold:
                    break
                results.append({
                    "index": index,
                    "text": self.tnt[0].corpus[index],
                    "distance": dist
                })
        elif return_distance:
            for index, dist, _ in nearset:
                if dist >= threshold:
                    break
                results.append({
                    "index": index,
                    "distance": dist
                })
        elif return_text:
            for index, dist, _ in nearset:
                if dist >= threshold:
                    break
                results.append({
                    "index": index,
                    "text": self.tnt[0].corpus[index],
                })
        else:
            for index, dist, _ in nearset:
                if dist >= threshold:
                    break
                results.append(index)


        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"T={elapsed_time:.4f}s")
        return results

    def save(self, filename):
        import dill
        with open(filename, "wb") as f:
            dill.dump(self, f)


def load(filename):
    import dill
    with open(filename, "rb") as f:
        tnt = dill.load(f)
    return tnt
