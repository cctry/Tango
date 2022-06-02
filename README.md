# SC22-Artifact
## Run docker
> docker build -t tango
> docker run -it tango
## Results reproduce
The scripts used for evaluation are in the directory 'evaluation'.
> cd evaluation
### Model speedup (Fig. 8)
 - GCN: python3 test_gcn.py --dataset=reddit
 - GAT: python3 test_gat.py --dataset=reddit
### Primitive speedup (Fig. 13-15)
- SPMM: python3 SPMM_broadcast.py|SPMM_nobroadcast.py
- SDDMM: python3 SDDMM_test.py
- GEMM: python3 linear_test.py
### Tested hardware
- GPU: NVIDIA V100S
- CPU: Intel(R) Xeon(R) Gold 6244 @ 3.60GHz
## Software requirements
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

