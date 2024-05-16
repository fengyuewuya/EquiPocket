# EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction
Welcome!
These files are the source code for our work. (EquiPocket: an E(3)-Equivariant Geometric Graph Neural Network for Ligand Binding Site Prediction)

## Files
 - protein\_feature.py: Extract the local geometric and global structure features in a protein, which needs MSMS to generate protein surface. 
 - models/
    - EquiPocket.py: Our proposed EquiPocket Framework. 
    - surface\_egnn.py: Our proposed Surface-Egnn model. 
    - baseline\_models.py: The implementation for some baseline models.  
    - egnn\_clean.py: The source code for "E(n) Equivariant Graph Neural Networks" from https://github.com/vgsatorras/egnn/tree/main/models. 

 - processed_data/
   - 5ei3\_protein.pdb: A clean protein from the PDBbind dataset. 
   - 5ei3\_protein.pkl: The processed graph data for 5ei3_protein. 

## Dependency
 - python: 3.7
 - cuda: 11.6
 - python packages: requirements.txt
 - MSMS: https://ccsb.scripps.edu/msms/


## Datasets
 - scPDB: http://bioinfo-pharma.u-strasbg.fr/scPDB
 - PDBbind: http://www.pdbbind.org.cn
 - COACH420: https://github.com/rdk/p2rank-datasets/tree/master/coach420
 - HOLO4K: https://github.com/rdk/p2rank-datasets/tree/master/holo4k




