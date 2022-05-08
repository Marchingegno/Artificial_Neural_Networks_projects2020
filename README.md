# Artificial Neural Networks coursework - Politecnico di Milano

## Tasks
1. Using a Neural Network classifier to ascertain wether all, some or no people in the pictures wear surgical masks.
2. Using a Neural Network to analyze crop pictures and identify what is dirt, plant or weed.
3. Using a Neural Network to answer question-image pairs.


## Solution
The deliveries are all named "Homework n" inside their respective folders. Solutions are implemented in **Python** using **Keras**, along with some **Matlab** code for preprocessing images.
Here a brief summary of the solutions:

### Homework 1
A **two steps approach**: the first step is to use a CNN to detect if there is at least one mask in the image and, if it detects at least one mask, the second step is to distinguish if there is at least a face without a mask. 
To avoid overfitting, we used data augmentation to rotate, shift, flip and zoom images.
The best approach is based on two VGG16 models, reaching a total accuracy of 86%.

### Homework 2
The crop images were very large, so we used **tiling** to split images into 256x256 tiles and process them separately, then combine them together into a single image. This reaches a better performance than simply resizing the image.

Since the network needs to distinguish between terrain and plant, and the first one is generally brown and the second is green, we performed some **preprocessing** on the images to increase contrast between background and plants. We used Matlab to perform this procedure. The transformation used for the enhancement is new_pixel_value = (red - green) / blue.

<img src="https://i.imgur.com/WkrWcA9.png" width=660 height=500></img>

The best model is the VGG encoder and its mirrored structure as decoder with l2 normalization, reaching 69% accuracy.

## Homework 3
We encoded the questions with a Keras tokenizer, after removing common words, then create a CustomDataset object that returns a traning batch composed of the encoded questions and corresponding images.
We used Xception as a base and customized it, merging the input from the question to the encoded image, reaching an accuracy of 62%.

