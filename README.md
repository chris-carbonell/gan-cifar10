# Overview

<b>gan-cifar10</b> trains a <b>generative adversarial network (GAN)</b> to generate images based on the CIFAR-10 dataset using <b>Keras</b>

# Why?

I worked through the GeeksforGeeks tutorial (link) to gain a better understanding of:
* <b>generative adversarial networks (GANs)</b>
* <b>Keras</b>

# GIF

The following GIF shows the evolution of the GAN over 15,000 epochs:
![evolution.gif](evolution.gif)

# Quickstart

To train the GAN from scratch, use:<br>
<code>python -m train</code>

# Table of Contents

| Path          | Description                       |
|---------------|-----------------------------------|
| model         | model H5 (generator.h5)           |
| evolution.gif | evolution of predictions by epoch |
| README.md     | documentation                     |
| train.py      | train a GAN                       |

# Installation

Install the required Python dependencies:<br>
<code>python -m pip install -r requirements.txt</code>

I'd recommend running this all in a virtual environment or a Docker container.

# Resources

* GeeksforGeeks Tutorial<br>
[https://www.geeksforgeeks.org/building-a-generative-adversarial-network-using-keras/](https://www.geeksforgeeks.org/building-a-generative-adversarial-network-using-keras/)
* Keras CIFAR10<br>
[https://keras.io/api/datasets/cifar10/](https://keras.io/api/datasets/cifar10/)
* CIFAR-10 Data<br>
[https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)