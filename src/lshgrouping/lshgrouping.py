import logging
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClusterMixin
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
from .indexdict import indexdict
from .unionfind import UnionFind


def array_to_minhash(arr: np.ndarray, num_perm: int = 128) -> MinHash:
    """
    Convert a binary array into a MinHash object.

    Parameters
    ----------
    arr : np.ndarray
        Binary fingerprint (1D array of 0/1 values).
    num_perm : int
        Number of hash permutations to use.

    Returns
    -------
    MinHash
        A datasketch MinHash representation of the input.
    """
    mh = MinHash(num_perm=num_perm)
    onbits, = np.where(arr)
    for bit in onbits:
        mh.update(str(bit).encode('utf8'))
    return mh


class LSHGrouping(ClusterMixin, BaseEstimator):
    """
    An implementation of locality-sensitive hashing (LSH) for clustering
    binary fingerprints using MinHash and Union-Find.

    Parameters
    ----------
    num_perm : int, default=128
        Number of permutations to use in MinHashing.
    lsh_threshold : float, default=0.8
        Similarity threshold used in LSH query.
    n_jobs : int, default=1
        Number of jobs to use for parallel operations. -1 means using all processors.
    verbosity : int, default=0
        Verbosity level: 0 (warnings), 1 (info), 2 (debug with progress bars).

    Attributes
    ----------
    minhashes_ : List[MinHash]
        Cached MinHash representations of the input.
    lsh_ : MinHashLSH
        Fitted LSH index.
    labels_ : np.ndarray
        Cluster labels for training data.
    """

    def __init__(
        self,
        num_perm: int = 128,
        lsh_threshold: float = .8,
        *,
        n_jobs: int = 1,
        verbosity: int = 0
    ):
        self.num_perm = num_perm
        self.lsh_threshold = lsh_threshold
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.minhashes_ = None
        self.lsh_ = None
        self.labels_ = None
        self.logger = self._setup_logger()
        

    def fit(self, X: np.ndarray, y=None) -> 'LSHGrouping':
        """
        Compute LSH-based clustering of binary input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_bits)
            Binary input data.
        y : Ignored
            Not used, present for API compatibility.

        Returns
        -------
        self : LSHGrouping
            Fitted instance.
        """
        n_samples = X.shape[0]
        self.logger.info(f"Fitting LSHGrouping to {n_samples} samples.")

        # Generate MinHashes
        self.logger.debug("Generating MinHashes...")
        task = (delayed(array_to_minhash)(x, num_perm=self.num_perm) for x in X)
        if self.verbosity >= 2:
            task = tqdm(task, desc='MinHashing', total=len(X))
        minhashes =  Parallel(n_jobs=self.n_jobs)(task)
        
        # Build LSH index
        self.logger.debug("Building LSH index...")
        lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        hashiter = minhashes if self.verbosity < 2 else tqdm(minhashes, desc='Indexing', total=len(minhashes))
        for i, mh in enumerate(hashiter):
            lsh.insert(str(i), mh)

        # Querying and connecting LSH neighbors
        self.logger.debug("Querying for LSH neighbors...")
        ufset = UnionFind(range(n_samples))
        hashiter = minhashes if self.verbosity < 2 else tqdm(minhashes, desc='Querying', total=len(minhashes))
        for i, mh in enumerate(hashiter):
            neighbors = lsh.query(mh)
            for j in map(int, neighbors):
                if i < j:
                    ufset.merge(i, j)

        # Assign cluster labels
        self.logger.debug("Assigning cluster labels...")
        label_map = indexdict()
        labels = [label_map[ufset[i]] for i in range(n_samples)]
        self.logger.info(f"Done. Identified {len(label_map)} clusters.")

        # Save results
        self.minhashes_ = minhashes
        self.lsh_ = lsh
        self.labels_ =  np.array(labels, dtype=np.int32)
        return self    

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Fit the model and return cluster labels for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_bits)
            Input binary data.

        y : Ignored
            Not used, for API consistency.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X, y)
        return self.labels_

    def _setup_logger(self) -> logging.Logger:
        """
        Create and configure a logger for this instance.

        Returns
        -------
        Logger
            Configured logger instance.
        """
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if self.verbosity < 1:
            logger.setLevel(logging.WARNING)
        elif self.verbosity < 2:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.DEBUG)
        return logger