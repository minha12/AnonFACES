# AnonFACES
---

With increasing camera surveillance systems everywhere and the rise of the autonomous cars, the concerns about privacy have never been addressed widely as much on the mainstream media like before. In this project, we are working on the problem of preserving privacy by concealing the facial identities in the multimedia database. It's the well-known problem of image-deidentification that has been researched since the early days of the Internet's widespread. However, with new technologies of Deep Learning and image processing, we revive the problem with a new approach that turns the image/video database with tens of GB or TB into a significantly condense dataset. Without a noticeable loss of utility, we can do different kinds of statistical analysis before applying k-Anonymity algorithm and optimize the information loss by state-of-the-art facial image synthesizes called styleGAN. We'll show that our methodology does not only have an advantage in providing a clear trade-off between privacy vs utility but also provide a higher level of privacy at the same requirement for utility. 
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
