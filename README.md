# Image-deidentification
---

With increasing camera surveillance systems everywhere and the rise of the autonomous cars, the concerns about privacy have never been addressed widely as much on the mainstream media like before. In this project, we are working on the problem of preserving privacy by concealing the facial identities in the multimedia database. It's the well-known problem of image-deidentification that has been researched since the early days of the Internet's widespread. However, with new technologies of Deep Learning and image processing, we revive the problem with a new approach that turns the image/video database with tens of GB or TB into a significantly condense dataset. Without a noticeable loss of utility, we can do different kinds of statistical analysis before applying k-Anonymity algorithm and optimize the information loss by state-of-the-art facial image synthesizes called styleGAN. We'll show that our methodology does not only have an advantage in providing a clear trade-off between privacy vs utility but also provide a higher level of privacy at the same requirement for utility. Furthermore, our surrogate faces have real looking which is ready for real-life applications.

<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/cluster.png'>
</p>  
<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/fakeid.png'>
</p>

---
## Branches

Please switch to your branch name before you make any change. I suggest to make branch names as below or similar ones to easily find conflict while merging:
- Minh-Ha Le (@minha12) - minhha
- Md Sakib Nizam Khan (@sakib570) - sakib
- Georgia Tsaloli (@tsaloligeorgia) - georgia

***


## Table of contents
Main directory:

| File/Folder | Description |
| --- | --- |
| GNN | Generative Neural Network for generating fake ID |
| clustering | Algorithms and evaluations for different clustering algorithms.|

In `clustering/` folder:
| File/folder | Description |
| --- | --- |
| `algorithms/` | Approximate Rank Order clustering, Constraint size K-Means, Outliers Handlers for DBSCAN & HDBSCAN, accuracy calulator |
| `encoding_data/` | Encoded data calculated by FaceNet or ResNet |
| `evaluation/` | Comparison of accuracy cross clustering algorithms, elbow method for K-Means & DBSCAN |
| `clustering.ipynb` | Jupyter notebook for testing clustering algorithms |

## How to use 

1. Approximate Rank Order Clustering
Run directly from the console without any input. This scripting file will take `encodings.pickle` as input and call `aroc.py` as clustering algorithm
```console
cd clustering/algorithms
python3 aprox_test.py --encodings ../encoding_data/encodings.pickle
```

2. Clustering Faces
Choosing one of the file `algorithms/cluster_faces.py` or `clustering.ipynb` for clustering the input in `encodings.pickle`. Currently, there are four clustering algorithms can be chosen (choose one of them and block the code for the others):
- HDBSCAN
- K-Means
- OPTICS
- DBSCAN
> Note that the `clustering.ipynb` cannot show the output of clusetring with a frame of faces for each cluseter, instead, look into the console output for the results. It will load the `encodings.pickle` by default if nothing is changed manually in the input data.

Running `cluster_faces.py`:
```console
python3 cluster_faces.py --encodings data/encodings.pickle
```
3. Encodings Faces
We can calculate encodings from different sources. 
```console
python3 encoding_faces.py --dataset folder_of_face_dataset --encodings data/ name_of_output_file.pickle
```
## Metric
Sum square distance provided in `clustering/algorithms/accuracy.py` is computed as below:
```
Step 1. Compute normalized square distance within cluster 
For each cluster:
  1. Compute square distances from each member to other members in the cluster
  2. Sum all the square distances
  2. Divide the sum by the size of the cluster
Step 2. Sum all the results in Step 1. Return this sum.
```
## Results:
Surrogate face analysis
<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/DNN_vs_PCA.png' width='400'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/styleGAN_vs_GNN.png' width='400'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/fakeID_to_orginal.png' width='400'>
</p>
Clustering analysis
<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/clustering-comparison.png' height='300'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/cluster_pairwise.png' height='300'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/cluster_silhouette.png' height='300'>
</p>
Interpolation between IDs
<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/alpha.png' height='300'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/dist_alpha.png' height='300'>
</p>

## Todos
1. Fixed-size clustering(done)
2. Clustering algorithm evaluation(done):
  + Calculate sum square distance (SSD) for all the clustering algorithm
  + Compare different clustering algorithms based on SSD
3. Fake ID evaluation (done):
  + Using trained GNN to synthesize face image for each cluster
  + Re-calculate the 128-D encoding
  + Calculate SSD from the output to all cluster members
4. Using FaceNet to calculate 128-D encodings for Celebs dataset (done)
5. Run both the clustering and Fake ID evaluation again with FaceNet encoding (__minhha__)
  + Clustering comparisons for Celebs dataset
  + Finding another dataset which can be used for Fake ID generation rather than RaFD: FEI Face Database
6. Show the trade-off between information loss (SSD) vs k value of k-Anonymity model
  + Look into implementations of related works on clustering: Direct distance, AAM, PCA dimention reduction, t-SNE (__sakib__)
  + Combine results from No. 5 and No.6 to show trade-offs and comparison to related works (__minhha__)
7. Using face recognition software to test re-identification (__sakib & georgia__)

## Next Goal:

__ESORICS 2020__ 

Important Dates and Deadlines
+ Title and Abstract deadline: April 3, 2020 (11:59 p.m. AoE - UTC-12)
+ Paper submission deadline: April 10, 2020 (11:59 p.m. AoE - UTC-12) 
+ Notification to authors: June 15, 2020 
+ Camera ready due: June 30, 2020 
+ Conference: September 14-18, 2020

Link: http://esorics2020.sccs.surrey.ac.uk/cfp.html
