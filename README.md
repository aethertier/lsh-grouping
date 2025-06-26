# lsh-grouping

An implementation of Locality-Sensitive Hashing (LSH) to cluster large, sparse, 
high-dimensional binary data. The interface is designed to be sklearn-compatible.


## Installation

```sh
# Clone repository source
git clone https://github.com/aethertier/lsh-grouping.git

# Enter repository directory
cd lsh-grouping

# Install the package
make install
```


## Usage

Below an example for clustering molecular fingerprints is given. The algorithm
works for any type of sparse binary features though. A more extensive example can be found in the [examples folder](https://github.com/aethertier/lsh-grouping/tree/main/examples).

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from lshgrouping import LSHGrouping

def smiles2fp(smi: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    fpgen = GetMorganGenerator(radius=3, fpSize=2048)
    return fpgen.GetFingerprintAsNumPy(mol)

# Input molecules (this could be millions of SMILES)
smiles = """\
CC1(C)c2cc(Br)ccc2N2CCC(O)=NC12C
Cc1c(CNC(=O)Cn2nc(cc2C(F)F)C(F)F)cnn1C
Cn1cc(CNC(=O)Cn2nc(cc2C(F)F)C(F)F)cn1
Cn1nccc1CNC(=O)Cn1nc(cc1C(F)F)C(F)F
CCN(Cc1ccc(OC)cc1)C(=O)c1cnc(OC)cc1C(F)(F)F
CCN(Cc1ccc(OC)cc1OC)C(=O)c1cnc(OC)cc1C(F)(F)F
CN1CCC(CC1)N1CCCC(C1)NC(=O)c1ccccc1Cl
CSc1ccc(CC(=O)NC2CCCN(C2)C2CCN(C)CC2)cc1
COc1ccc(C(=O)NC2CCCN(C2)C2CCN(C)CC2)c(OC)c1
COc1ccc(cc1OC)C(=O)NC1CCCN(C1)C1CCN(C)CC1""".split('\n')

# Generate fingerprints from SMILES (N, 2048)
X = list(map(smiles2fp, smiles))

# Cluster fingerprints
lsh = LSHGrouping(
    num_perm=128, 
    lsh_threshold=.8, 
    verbosity=2, 
    n_jobs=4)
clust_assign = lsh.fit_predict(X)
```