## Table Of Contents
- [Introduction](#introduction)
- [1. Data Analysis](#data-analysis)
    -[a. Removing an outlier](#removing-an-outlier)
    -[b. Highest turnover in 2016](#highest-turnover-in-2016)
  -[c. Top 5 weeks for department 88 in 2015](#top-5-weeks-for-department-88-in-2015)
  -[d. Top performer store of 2014](#top-performer-store-in-2014)
  -[e. What kind of sport is department 73](#what-kind-of-sport-is-department-73)
  -[f. What kind of sport is department 117](#what-kind-of-sport-is-department-117)
  -[g. A representation of business unit features](#a-representation-of-business-unit-features)
  -[h. An overall vue on data](#an-overall-vue-on-data)
- [2. Modeling](#modeling)
  -[a. Firsts models](#firsts-models)
  -[b. Final model](#final-model)
- [3. Datasets](#datasets)
- [4. Final model architecture](#final-model-architecture)
- [5. Results](#results)
- [6. Pipeline](#pipeline)
- [7. Using pipeline](#using-pipeline)
- [8. Conclusion](#conclusion)
- [References](#references)

## Introduction

This project aims at developing a machine learning model able to predict from 1 to 8 next weeks turnovers for a given store and department.

## Data Analysis

## Try my models

Check and try the <a href="Train.ipynb" target="_blank">Notebook</a> that I used to train my models : <a href="https://colab.research.google.com/github/Edouard99/GAN/blob/main/Train.ipynb" target="_blank">here on Google Colaboratory</a>

## Dataset

In this project I have used different datasets :
* A pokemon dataset (with data augmentation in 64px) that you can find <a href="https://drive.google.com/file/d/1mKWPRvdYg6jfN6G8AFxsHpzjo3608QaJ/view">here</a>.
* A pokemon dataset (with data augmentation in 256px) that you can find <a href="https://drive.google.com/file/d/19sQKN9H4gmNPxQLLtjDmw5SV5KZ4Q1n1/view">here</a>.

## Special Gan Features 

###  Mini Batch Discrimination

The following networks are mostly convolutional and deconvolutional networks. Yet it is known that GANs are based on complex equilibrium and that they can quickly become unstable and go into collapse mode :
<p align="center">
  <img alt="Collapse Mode" title="Collapse Mode" src="./Media/collapse.PNG" >
</p>
In both DCGAN and WGAN models a Mini Batch Discrimination Layer is implemented in the Discriminant. This layer prevents the Generator to produce the same image for multiple inputs. For more informations see <a href="https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b">this link</a>.

### Label Smoothing

To avoid overconfidence for the Discriminant in the DCGAN model a BCE with label smoothing is used. See the custom function in <a href="./Trainings/basic_functions.py">basic_functions.py</a>.

### Noisy Labels

Once again to avoid overconfidence in the Discriminant, in one of the trainings of the DCGAN model, the labels are swapped between real images and fake, see in <a href="./Trainings/training_boosting.py">training_boosting.py</a>.

## DC-GAN 


### DC-GAN Architecture

A Deep Convolutional GAN (DC-GAN) is developed for (64px and 256px) using <a href="./Models/gan_64.py">gan_64.py</a> and <a href="./Models/gan_256.py">gan_256.py</a> with the mode "dcgan".
The architecture of the GAN is given by the following figures :

<p align="center">
  <img alt="Generator DCGAN" title="Generator DCGAN" src="./Media/Generator.PNG" >
</p>
<p align="center">
  <img alt="Discriminant DCGAN" title="Discriminant DCGAN" src="./Media/Discriminant_DC.PNG" >
</p>
<p align="center">
  <img alt="Training DCGAN" title="Training DCGAN" src="./Media/DCtraining.PNG" >
</p>

The generator input is an nz-sized vector (noise) that will be deconvoluted into an image.
The discriminant's input is an image and the output is a number $\in$ [0;1]. 1 means that the image is considered as being real and 0 means that it is considered as being fake.

### Classic Training

The first training used is a classic DC GAN training, defined in <a href="./Trainings/training_classic.py">training_classic.py</a>. The discriminant D is fed with real (x) and fake (G(z)) images of the generator G at each iteration and the loss is calculated with a Binary Cross Entropy loss function :

$$ D_{Loss}(x,z)= -(log(D(x)) + log(1-D(G(z))))$$

$$ G_{Loss}(z)= -(log(D(G(z))))$$

Then discriminant and generator are optimized to minimize these losses.
An Adam optimizer (lr=0.00015, beta1=0.5, beta2=0.999) has been used for both networks.

### Monitored Training
The second training used is a classic DC GAN training with a monitoring of the loss values that influences the training, defined in <a href="./Trainings/training_monitoring.py">training_monitoring.py</a>. The discriminant and generator loss are defined as for a classic training. A threshold is defined such as at each iteration the discriminant is optimized only if :

$$D_{Loss}>0.8*G_{Loss}$$

This is done to prevent the discriminant to become too powerful compared to the generator.
An Adam optimizer (lr=0.00015, beta1=0.5, beta2=0.999) has been used for both networks.

### Boost and Noisy Training
The third training used is a DC GAN training defined in <a href="./Trainings/training_boosting.py">training_boosting.py</a>. At each iteration, a random number k (uniform distribution) is computed and this number defines what will be trained during this iteration and how.

* if $0.0001 \lt k \lt 0.001$ for the next 100 iterations(including this one) ONLY the discriminant will be trained with real images labeled as real and fake images labeled as fake. This results in a boost in the training of the discriminant.
* if $0.001 \lt k \lt 0.93$ this iteration the discriminant will be trained with real images labeled as real and fake images labeled as fake.
* if $0.93 \le k \le 1$ this iteration the discriminant will be trained with real images labeled as fake and fake images labeled as real (Noisy Label) in order to add noise in the training for a more robust discriminant.
* if $0 \le k \lt 0.0001$ for the next 100 iteration(including this one) ONLY the generator will be trained. This results in a boost in the training of the generator.
* if $0.001 \lt k \le 1$ for this iteration the generator will be trained.

An Adam optimizer (lr=0.00015, beta1=0.5, beta2=0.999) has been used for both networks.

## W-GAN
### W-GAN Architecture

A Wasserstein GAN (W-GAN) is developed for (64px and 256px) using <a href="./Models/gan_64.py">gan_64.py</a> and <a href="./Models/gan_256.py">gan_256.py</a> with the mode "wgan".
The architecture of the GAN is given by the following figures :

<p align="center">
  <img alt="Generator W-GAN" title="Generator W-GAN" src="./Media/Generator.PNG" >
</p>
<p align="center">
  <img alt="Discriminant W-GAN" title="Discriminant W-GAN" src="./Media/Discriminant_WGAN.PNG" >
</p>
<p align="center">
  <img alt="W-GAN training" title="W-GAN training" src="./Media/WGANtraining.PNG" >
</p>

The generator input is a nz-sized vector (noise) that will be deconvoluted into an image.
The discriminant's input is an image and the output is a number $\in \mathbb{R}$. This number is a score that assesses the realness of the image.

### W-GAN Training

The training is a Wasserstein GAN training defined in <a href="./Trainings/training_wgan.py">training_wgan.py</a>.
The discriminant D is fed with real (x) and fake (G(z)) images of the generator G at each iteration and the loss is calculated with a Binary Cross Entropy loss function :

$$ Distance_{D} = D_{Loss}(x,z) = (\sum_{i=1}^{batch size} D(x_{i}) - \sum_{j=1}^{batch size} D(G(z_{j})))$$

$$ Distance_{G} = G_{Loss}(z)= -(\sum_{i=1}^{batch size} D(G(z_{i})))$$

The goal of the discriminant is to maximize the distance $Distance_{D}$ between the distributions of the score given for real images and the distribution of the score given for fake images. In this project, we choose to minimize $-Distance_{D}$. The more the distribution will be separated the more the decision to assess if the image is real or fake is accurate.
The goal of the generator is to minimize the distance $Distance_{G}$ between the distributions of the score given for real images and the distribution of the score given for fake images. The generator tries to fool the discriminant with generated images that should be classified as real, it is reducing the distance between the distributions.
During this training, we apply weights clipping on the discriminant's weights and the discriminant is trained 3 times more than the generator.

An RMSProp optimizer (lr=5e-5) has been used for both networks for stability reasons.

For more informations on WGAN training check <a href="https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/">this website</a>.

## Results

Here are some results of the networks :

### DC-GAN classic

<p align="center">
  <img alt="Classic DCGAN gif" title="Classic DCGAN gif" src="./Media/animpokemon64_dcgan_classic_150.gif" width="450">
</p>

<p align="center">
  <img alt="Classic DCGAN last" title="Classic DCGAN last" src="./Media/pokemon64_dcgan_classic_150.png" width="450">
</p>

### DC-GAN monitoring

<p align="center">
  <img alt="Monitoring DCGAN gif" title="Monitoring DCGAN gif" src="./Media/animpokemon64_dcgan_monitoring_150.gif" width="450">
</p>

<p align="center">
  <img alt="Monitoring DCGAN last" title="Monitoring DCGAN last" src="./Media/pokemon64_dcgan_monitoring_150.png" width="450">
</p>

### DC-GAN boosting and noisy label

<p align="center">
  <img alt="Boost DCGAN gif" title="Boost DCGAN gif" src="./Media/animpokemon64_dcgan_boosting_150.gif" width="450">
</p>

<p align="center">
  <img alt="Boost DCGAN last" title="Boost DCGAN last" src="./Media/pokemon64_dcgan_boosting_150.png" width="450">
</p>

### W-GAN

<p align="center">
  <img alt="WGAN gif" title="WGAN gif" src="./Media/animpokemon64_wgan_150.gif" width="450">
</p>

<p align="center">
  <img alt="WGAN last" title="WGAN last" src="./Media/pokemon64_wgan_150.png" width="450">
</p>

## Conclusion

Once trained, the 64px networks produce images with sharp edges like small terrifying monsters.

Moreover the Mini Bacth Discrimination layer seems to perform well : even images that looks close have some variations (same type of pokemon but still a bit different). Even with a small latent space the networks always produce different outputs.

An improvement of these 64px networks is planned for the future.

For now, I have trained the 256px networks and obtained good results (images with sharp edges). Yet I believe that I need to tune the hyperparameters to get better results. These networks takes a lot of time to train so I will update this project as soon as I will have the time to compute good results !
