# AnonFACES: Anonymizing Faces Adjusted to Constraints onEfficacy and Security

---

This is official implementation of paper 'AnonFACES: Anonymizing Faces Adjusted to Constraints onEfficacy and Security'

---

With increasing camera surveillance systems everywhere and the rise of the autonomous cars, the concerns about privacy have never been addressed widely as much on the mainstream media like before. In this project, we are working on the problem of preserving privacy by concealing the facial identities in the multimedia database. It's known as problem of image-deidentification. However, with new technologies of Generative Adversarial Network (GAN) and Convolutional Neural Network (CNN), we revive the problem with a new approach that turns the image/video database with tens of GB or TB into a significantly condense latent space and embbeded face descriptors. Without a loss of generality, we can do different kinds of statistical analysis before applying k-Anonymity algorithm and optimize the information loss by state-of-the-art facial image synthesizes called styleGAN. We'll show that our methodology does not only have an advantage in providing a clear trade-off between privacy vs utility but also provide a higher level of privacy at the same requirement for utility. 
<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/cluster.png'>
</p>  
<p align='center'>
  <img src='https://github.com/minha12/image-deidentification/blob/minhha/clustering/evaluation/img/fakeid.png'>
</p>

---

# Preliminary
Before you start the notebooks:
- Make sure required packages in ```requirements.txt``` are installed (please note that we ran our experiments on Microsoft's Azure Data Science Virtual Machine with some pre-installed packages, if you run our code on your local machine, the requirements may be differ)
- Download a pre-trained StyleGAN model ```karras2019stylegan-ffhq-1024x1024.pkl``` (can be found at [StyleGAN's Github repo](https://github.com/NVlabs/stylegan) ) and put it to folder ```stylegan/cache/```
- Download our [pre-trained CNN model](https://drive.google.com/file/d/1EhaiYQ0uWPPkmglnwjUNX_m6WU92z_bL/view?usp=sharing) ```FaceGen.RaFD.model.d5.adam.h5``` and put it to folder ```GNN/output/```
- Prepare your facial descriptors (embbedings) which can be calculated by [Dlib](http://dlib.net), [FaceNet](https://github.com/davidsandberg/facenet) or PCA. Default folder is ```datasets/encoding_data/  
- Prepare your StyleGAN's latent vectors which can be calculated by [StyleGAN-encoder](https://github.com/Puzer/stylegan-encoder). Default folder is ```datasets/stylegan_data/latent_vectors```. 

---
# Paths


|Path | Description
| :--- | :---
├── CNN | Submodule of Up-Convolutional Network
├── README.md
├── [NB]\ NumericalExperiments.ipynb | Evaluations for face descriptors/anonymisers
├── [NB]\ Partitioning.ipynb | Evaluation of fixed-size clustering (partitioning) algorithm
├── [NB]\ VisualizingResults.ipynb | Visualising data
├── anonymizer | Image anonymiser/syntheriser
│   ├── __init__.py
│   ├── aam.py | Active Appearance Model 
│   ├── cnn.py | Up-convolutional Network
│   └── styleGan.py | StyleGAN
├── datasets | Prepared data for evaluations
│   ├── encoding_data | Face desciptors / embbeddings
│   └── stylegan_data | Encoded latent vectors
├── evaluation.py | Support functions for evaluations
├── outputs | Folder for outputs
├── partitioning.py | Fixed-size clustering algorithms
├── requirements.txt
├── stylegan | Submodule of StyleGAN
├── utils.py | Utilities functions
└── vizualization.py | Support functions for visualising results


---
# Evaluations

The evaluations for the paper is provided in the notebooks:
- Evaluation with different face embeddings, anonymizers ```NumericalExperiments.ipynb```
- Evaluation with different fixed-size clustering algorithms ```Partitioning.ipynb```
- Visualizing the evaluation results ```VisualizingResults.ipynb```
