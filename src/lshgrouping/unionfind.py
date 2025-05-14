"""
Implementation of Union-Find 

The syntax is intended to match SciPy's DisjointSet, though 
some functionalities are missing for improved  performance.
"""
from typing import Hashable, Iterable


class UnionFind:
    """Data structure for incremental connectivity queries. 

    Methods
    -------
    __getitem__
    add
    merge
    connected
    """

    def __init__(self, elements: Iterable[Hashable]):
        self._parent = {elem: elem for elem in elements}
        self._size = {elem: 1 for elem in elements}

    def __len__(self):
        return len(self._parent)

    def __contains__(self, element: Hashable):
        return element in self._parent
    
    def __getitem__(self, element: Hashable):
        """Find the root `element`.

        Parameters
        ----------
        element : Hashable object
            Input element.

        Returns
        -------
        root : Hashable object
            Root `element`.
        """
        return self._find(element)

    def _find(self, element: Hashable) -> Hashable:
        """Find the root `element`. Compresses the path, if possible.

        Parameters
        ----------
        element : Hashable object
            Input element.

        Returns
        -------
        root : Hashable object
            Root `element`.
        """
        # Find root
        root = self._parent[element]
        n = 0
        while self._parent[root] != root:
            root = self._parent[root]
            n += 1
        # Compress path if needed
        if n > 0: self._compress_path(element, root)
        # Return root
        return root

    def _compress_path(self, element: Hashable, root: Hashable):
        """Compress the path from `element` to its root."""
        while element != root:
            self._parent[element], element  = root, self._parent[element]
        
    def add(self, new_element: Hashable):
        if new_element in self:
            raise ValueError(f"Cannot add existing element: '{new_element}'")
        self._parent[new_element] = new_element
        self._size[new_element] = 1

    def connected(self, element_x: Hashable, element_y: Hashable):
        """Test whether `element_x` and `element_y` are in the same subset."""
        root_x, root_y = self._find(element_x), self._find(element_y)
        return root_x == root_y

    def get_set_size(self, element: Hashable) -> int:
        """Returns the size of the subset containing `element`."""
        root = self[element]
        return self._size[root]

    def merge(self, element_x: Hashable, element_y: Hashable):
        """Union the sets containing `element_x` and `element_y`, using size heuristic."""
        root_x, root_y = self._find(element_x), self._find(element_y)
        size_x, size_y = self._size[root_x], self._size[root_y]
        if root_x == root_y:
            return  # Already in the same set
        if size_x > size_y:
            self._parent[root_y] = root_x
            self._size[root_x] += size_y
        else:
            self._parent[root_x] = root_y
            self._size[root_y] += size_x