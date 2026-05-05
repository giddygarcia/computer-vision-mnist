# MNIST Digit Recognizer
## Deep Learning: Classifying with Neural Networks (MLPs and CNNs)

## Overview
The MNIST data can be used to learn computer vision fundamentals and deep learning foundations. Such foundations will be presented by **exploring the performance of MLPs and CNNs on this famous dataset.** [Dataset from Kaggle](https://www.kaggle.com/competitions/digit-recognizer)

## 🎯 Objectives
* Create a deep MLP Model
* Create a CNN Model
* Optimize different hyperparamters for both types of models 
* Pick the best model and observe its performance on the data.
* Sample predictions: Present its capabilities on samples of the data

## Key Findings
* When going from classification with Logistic Regression to Deep Learning, neural networks display an advantageous gap in performance: from ~91% accuracy to ~97%.
* Tuning hyperparamaters was always beneficial for the neural networks' accuracy.
* All models' final validation scores are:
    1. **Tuned CNN               -    0.9936**
    2. CNN Baseline            -    0.9814
    3. Tuned MLP               -    0.9771
    4. Base MLP                -    0.9664
    5. Log Regression          -    0.9137
* 🏆️ The best model is a **tuned CNN model reaching ~99.4% accuracy** on the test set.

## 📦 Packages and Libraries Used:
- torch
- pytorch-lightning
- lightning
- torchmetrics
- optuna
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

*A `requirements.txt` file is also available in this repo.*

### Viewing / Installation:
1. *Viewing Option:* Simply view the notebook file `mnist-digit-recognizer.ipynb`
2. *Full Installation Option:* Download the repository `git clone https://github.com/giddygarcia/computer-vision-mnist.git`


## ✉️ Author and Contact Information
Developed by: Christine Garcia 

Have questions? Feel free to:
* email me at cavgarcia22@gmail.com 
* connect on [LinkedIn](www.linkedin.com/in/cavgarcia) 
* or [view more projects](https://github.com/giddygarcia) that I enjoyed making