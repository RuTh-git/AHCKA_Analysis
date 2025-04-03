# AHCKA_Analysis

[Link to the paper](https://dl.acm.org/doi/abs/10.1145/3589261)

CITATIONS:

    @article{LiYS23,
      author       = {Yiran Li and
                      Renchi Yang and
                      Jieming Shi},
      title        = {Efficient and Effective Attributed Hypergraph Clustering via K-Nearest
                      Neighbor Augmentation},
      journal      = {Proc. {ACM} Manag. Data},
      volume       = {1},
      number       = {2},
      pages        = {116:1--116:23},
      year         = {2023}
    }

# HNCUT Commands:
For converting the dataset to HNCut Compatible dataset, use the below command:
```
python convert_to_hncut.py --input_root data --output_root hncut_data --subdir npz/20news
```

To run the Hncut on the converted dataset, use the below command:
```
python hncut.py --data hncut_data --dataset npz/20news
```

# AHCKA Commands:
To run AHCKA on the dataset, use the below command:
```
python ahcka.py --data coauthorship --dataset cora
```
# Sensitivity of AHCKA:
```
python AHCKA.py --data coauthorship --dataset dblp --sensitivity
```
