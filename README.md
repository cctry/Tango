# SC22-Artifact
## Run docker
> docker build -t tango
> docker run -it tango
## Results reproduce
The scripts used for evaluation are in the directory 'evaluation'.
> cd evaluation

The datasets are described in the below Datasets section. To specifiy the dataset used in expriments, add the dataset's name (in lowercase) as the option. The following commands use Reddit as an example. Other options have default values.
### Model speedup 
#### Node classfication (Fig. 8)
 - GCN: python3 test_gcn.py --dataset=reddit
 - GAT: python3 test_gat.py --dataset=reddit
 - HGT: python3 test_HGT.py --dataset=reddit
#### Link prediction (Fig. 9)
 - GCN: python3 link_pred_gcn.py --dataset=reddit
 - GAT: python3 link_pred_gat.py --dataset=reddit
 - HGT: python3 link_pred_HGT.py --dataset=reddit
### Framework-level optimization (Fig. 12-13)
python3 test_cache_GEMM.py --dataset=reddit
python3 test_fuse.py --dataset=reddit
### Primitive speedup (Fig. 14-19)
- SPMM: python3 SPMM_broadcast.py|SPMM_nobroadcast.py|test_incidence.py --dataset=reddit
- SDDMM: python3 SDDMM_test.py --dataset=reddit
- GEMM: python3 linear_test.py --dataset=reddit
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

