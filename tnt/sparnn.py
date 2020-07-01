from collections import deque
import numpy as np
import random
import math
import time


class Sparnn():
    def __init__(self, depth=2, verbose=True):
        self.leaders = {}
        self.indices = {}
        self.depth = depth
        self.verbose = verbose

    def search(self, vector, k=10, return_distance=False):
        start_time = time.time()
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector.todense()).flatten()

        level = (0,)
        depth = self.depth
        queue = []
        for i in range(depth - 1):
            leaders = self.leaders[level]
            similarity = leaders.dot(vector)
            best_leaders = similarity.argsort()[::-1]
            similarity.sort()
            if i < depth - 2:
                next_level = best_leaders[0]
                level = level + (next_level,)
            else:
                for candidate in best_leaders:
                    queue.append(level + (candidate,))
        
        indices = []
        distances = []
        num_candidates = 0
        for level in queue:
            level_candidates = self.leaders[level]
            level_indices = self.indices[level]
            num_candidates += level_indices.shape[0]

            similarity = level_candidates.dot(vector)
            distances.append(1 - similarity)

            indices.append(level_indices)
            if num_candidates >= k:
                break
        indices = np.concatenate(indices)
        distances = np.concatenate(distances)
        top_k = indices[distances.argsort()[:k]]

        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"T={elapsed_time:.4f}s")

        if return_distance:
            distances.sort()
            distances = distances[:k]
            return distances, top_k
        return top_k

    def fit(self, space):
        indices = np.arange(space.shape[0])
        self.traverse(space, indices)

    def traverse(self, space, indices, level=(0,)):
        N = space.shape[0]
        if len(level) >= self.depth:
            self.indices[level] = indices
            self.leaders[level] = space
            return

        # choose leaders
        n = math.ceil(int(N**.5))
        leaders = random.choices(list(range(N)), k=n)
        lead_vectors = space[leaders].T
        similarity = space.dot(lead_vectors)

        # get closest leader
        closest_leader = np.array(
            similarity.argmax(axis=1)
        ).flatten()

        # in case of indeterminacy, put vector in a random bucket
        max_dist = similarity.max(axis=1).todense()
        zero_ind = np.where(max_dist == 0)[0]
        closest_leader[zero_ind] = np.random.randint(
            0, n, size=(zero_ind.shape[0],))

        # store leaders at 
        self.leaders[level] = lead_vectors.T

        for i in range(n):
            ith_cluster = np.where(closest_leader == i)[0]
            if ith_cluster.shape[0] == 0:
                continue
            # print(level + (i,), indices.shape[0])
            self.traverse(
                space[ith_cluster],
                indices[ith_cluster],
                level=level + (i,))
