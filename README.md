# SC22-Artifact
## Run docker
> docker build -t tango
> docker run -it tango
## Softward requirements
- [DGL](https://github.com/dmlc/dgl) compiled from source with CUDA support
- [CUTLASS](https://github.com/NVIDIA/cutlass) 
- Pytorch 1.10.0
- CUDA 11.4
- GCC 7.5.0
## Datasets
- **ogbn-arxiv**: install ogb Python liraries as shown in [this link](https://ogb.stanford.edu/docs/nodeprop/).
- **Pubmed** and **Reddit** datasets are available in the latest version of [DGL](https://github.com/dmlc/dgl).
- **DBLP**: Extract the text file from [SNAP](https://snap.stanford.edu/data/com-DBLP.html)
- **Amazon**: Extract the text file from [SNAP](https://snap.stanford.edu/data/amazon0505.html)

All dataset needs to be put in the 'Dataset' folder, which may be created by ogb.
## Configuration
The CUDA source is compiled at runtime by Pytorch. The compile options are set in **cuda/kernels.py**.
The location of CUTLASS needs to be specified in this file as
> CUTLASS_PATH = '/path/to/cutlass'

