[![GitHub issues](https://img.shields.io/github/issues/arjunparmar/COVID-19?style=for-the-badge)](https://github.com/arjunparmar/COVID-19/issues) [![GitHub forks](https://img.shields.io/github/forks/arjunparmar/COVID-19?style=for-the-badge)](https://github.com/arjunparmar/COVID-19/network) [![GitHub stars](https://img.shields.io/github/stars/arjunparmar/COVID-19?style=for-the-badge)](https://github.com/arjunparmar/COVID-19/stargazers) [![GitHub license](https://img.shields.io/github/license/arjunparmar/COVID-19?style=for-the-badge)](https://github.com/arjunparmar/COVID-19/blob/master/LICENSE.md)
# COVID-19
COVID-19 (coronavirus disease 2019) is an infectious disease caused by **severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2)**, previously known as 2019 novel coronavirus (2019-nCoV), a strain of coronavirus. The first cases were seen in Wuhan, China, in late December 2019 before spreading globally. The current outbreak was officially recognized as a pandemic on **11 March 2020**.No effective treatment or vaccine exists currently (April 2020).
# DESCRIPTION
Since reverse transcription polymerase chain reaction (RT-PCR) test kits are in limited supply, there exists a need to explore alternative means of identifying and prioritizing suspected cases of COVID-19. In our City of Gujarat, Surat, turnaround time for COVID-19 test results is currently reported as 24 hours. It is possible that this turnaround time is even greater in rural and remote regions.
Moreover, large scale implementation of the COVID-19 tests which are extremely expensive cannot be afforded by many of the developing & underdeveloped countries hence if we can have some parallel diagnosis/testing procedures using Artificial Intelligence & Machine Learning and leveraging the historical data, it will be extremely helpful. This can also help in the process to select the ones to be tested primarily.
## Why Study X-Rays?
CT scans, despite being used to contribute to the clinical workup of patients suspected for COVID-19, are costly. Therefore, smaller centres may have limited access to CT scanners. X-ray machines are more affordable and can be portable and therefore a viable alternative.
To be clear, We don't claim that CXRs should be relied on in the diagnosis of COVID-19. Direction from the Canadian Association of Radiologists recommend following the CDC’s guidelines that state that CXR and CT should not be used to diagnose COVID-19. The recommendations state that viral testing is currently the only means to diagnose COVID-19, even if a patient has suggestive features on CXR or CT. That being said, some research has suggested that radiological imaging (CT in particular) may be just as sensitive as RT-PCR. Such findings exhibit why we’ve decided to explore this avenue and build open source tools to enable further investigation. It is also important to note that normal imaging does not imply that a patient has not been infected by SARS-CoV-2, only that they aren’t showing signs of COVID-19 respiratory disease.
## Goal
Determine if a machine learning classifier can be trained to distinguish cases of COVID-19 from CXRs.
# DATA
## DATASET
For the purpose of this experiment, data was taken from two repositories:<br/>
1)The [COVID-19 image data collection](https://github.com/ieee8023/covid-chestxray-dataset) repository on GitHub is a growing collection of deidentified CXRs and CT scans from COVID-19 cases internationally. We would like to take this opportunity to thank Joseph Paul Cohen and his fellow collaborators at the University of Montreal for their hard work in assembling this dataset. See Figure 1a for an sample image.<br/>
2)The [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) available on Kaggle contains several deidentified CXRs and includes a label indicating whether the image shows evidence of pneumonia. We would like to take this opportunity to thank the Paul Mooney Developer Advocate at KaggleBoulder, United States and all other involved entities for creating this dataset. See Figure 1b for an sample image.<br/>
|![](https://github.com/arjunparmar/COVID-19/blob/master/Data/Images/Positive.jpg)|![](https://github.com/arjunparmar/COVID-19/blob/master/Data/Images/Negative.jpg)|
|:---: | :---: |
|Figure 1a|Figure 1b|<br/>
## Data Preprocessing
Prior to training, preprocessing was implemented on the images themselves. ImageDataGenerator (from tensorflow.keras) was employed to perform preprocessing of image batches prior to training. The following transformations were applied to images:<br/>
Images were resized to have the following shape: 500×500×3. By reducing the image size, the number of parameters in the neural network was decreased.<br/>
images are normalized by scaling them so their pixel values are in the range [0, 1]
## Binary Classification
We first considered a binary classification problem where the goal was to detect whether an X-ray shows evidence of COVID-19 infection. The classifier was to assign X-ray images to either a non-COVID-19 class or a COVID-19 class. A deep convolutional neural network architecture was trained to perform binary classification. The model was trained using the RMSprop optimizer and binary cross-entropy loss. All training code was written using TensorFlow 1.15.
# Architecture
## Densenet
The name refers to Densely connected convolutional network, written by Gao Huang, Zhuang Liu, Laurens van der Maaten and Kilian Q. Weinberger, the paper garnered around 700 citations.<br/>
By this time around, Deep learning kind of started as replicating human vision has turned into an engineering practice, people are trying different things looking at the accuracies, network weights and visualizing features. There was a saying around this period, that NIPS conference paper acceptance has been degraded. Researchers are keeping lot of effort in engineering these networks instead of trying something different. Having said all of this, this paper is still a good read and has improved the accuracy of image classification challenges.<br/>
DenseNet is very similar to ResNet with two important changes.<br/>
1)Instead of adding up the features as in ResNet, they concat the feature maps.<br/>
2)Instead of just adding one skip connection, add the skip connection from every previous layer. Meaning, In a ResNet architecture, we add up (l-1) and (1) layer features. In DenseNet, for Lth layer we concat all the features from [1….(l-1)] layers.<br/>
The author quotes “DenseNet layers are very narrow (eg 12 filters per layer) adding only a small set of feature maps to the “collective knowledge” of the network and keep the remaining feature maps unchanged- and the final classifier makes a decision based on all feature-maps in the network.”<br/>
![Densenet](https://github.com/arjunparmar/COVID-19/blob/master/Data/Images/Densenet1.png)<br/>
Each block contains a set of conv blocks (Called bottleneck layer if contains a 1*1 conv layer to reduce features followed by 3*3 conv layers)depending on the architecture depth. The following graph shows DenseNet-121, DenseNet-169, DenseNet-201, DenseNet-264.<br/>
![Densenet](https://github.com/arjunparmar/COVID-19/blob/master/Data/Images/Densenet2.png)<br/>
Growth rate defines the number of features each dense block will began with. The above diagram uses 32 features. Having this kind of architectures reduces the parameters we train by a lot. For example a resnet conv block may contain [96, 96, 96] feature maps inside a block, where as DenseNet contains [32, 64, 96]. The author claims and has been proven by some other paper called stochastic depth network (where they train a ResNet architecture by dropping a few layers randomly every iteration) that most of the features learned by ResNet are redundant. The author basic intuition is also that connecting layers in this way where each layer contains the features of all the previous layers, will not allow it it learn redundant features.<br/>
The transition layers in the network reduces the number of features to \theta x m, where \theta takes values between (0, 1). \theta x m is also used in bottleneck layers. when \theta <0 for both transition layers and bottleneck layers it is called DenseNet-BC, when \theta <0 for only transition layers it is called DenseNet-C.<br/>
![Densenet](https://github.com/arjunparmar/COVID-19/blob/master/Data/Images/Densenet.png)<br/>
## Results on Architecture:
--It uses 3x less parameters compared to ResNet for similar number of layers.<br/>
--Using the same set of parameters used for ResNet architectures and replacing the bottleneck layers of resnet with Dense blocks, the authors have seen similar performance on ImageNet dataset. On CIFAR-10, CIFAR-100 and other datasets, DenseNet blocks have shown incremental performance.
[Paper on Densenet](https://arxiv.org/pdf/1608.06993.pdf)
# Measures Against Overfitting
We applied multiple strategies to combat overfitting, such as dropout regularization and data augmentation (varrying brightness of image by +10% and -10%).
# Class Imbalance
Due to the scarcity of publicly available CXRs of severe COVID-19 cases, we were compelled to apply class imbalancing methods to mitigate the effects of having one class outweighing several others. If you are versed in machine learning, you know that accuracy can be misleadingly high in classification problems when a class is underrepresented. We were left with a dilemma, as we also did not want to limit our training data to only 189 non-COVID-19 examples, as there would have been about 400 images total.As above said, we apply Image Augmentation to overcome this issue.
# Result
|Test Acc:99.33%|Loss:0.0123|
|:---:|:---:|
|![](https://github.com/arjunparmar/COVID-19/blob/master/Data/accuracy.png)|![](https://github.com/arjunparmar/COVID-19/blob/master/Data/loss.png)|<br/>
# Conclusion
|Test Accuracy|99.33%|
|:---:|:---:|
|**Loss**|**0.0123**|
|**F1 Score**|**0.55**|<br/>
# Install Dependencies
Ubuntu 16.10 and 17.04 do not come with Python 3.6 by default, but it is in the Universe repository. You should be able to install it with the following commands:<br/>
```$ sudo apt-get install python3.6```<br/>
Install package python-opencv with following command in terminal (as root user).<br/>
```$ sudo apt-get install python-opencv```<br/>
Install TensorFlow with Python's pip package manager.</br>
```pip install tensorflow==1.15```<br/>
you can install Matplotlib and all its dependencies with from the Terminal command line:<br/>
```pip install matplotlib```<br/>
# Deploy a Machine Learning model using Django
## What is the Django REST Framework?
Django is a high-level Python Web Development framework that encourages rapid development and clean, pragmatic design. It has been built by experienced developers, and takes care of much of the hassle of Web development. It is also free and open source.<br/>
Django REST Framework is a powerful and flexible toolkit for building Web APIs which can be used to Machine Learning model deployment. With the help of Django REST framework, complex machine learning models can be easily used just by calling an API endpoint.<br/>
## Installation
Django can be installed using a simple pip install.<br/>
```$ pip install django```<br/>
```$ pip install djangorestframework```<br/>
## Let’s Deploy!
### 1. Create a Django Project<br/>
To start a new Django project, we first need to take care of some initial setup. Namely, we will need to auto-generate some code that establishes a Django project — a collection of settings for an instance of Django, including database configuration, Django-specific options and application-specific settings. To start a new project, $ cd into the directory where you want to create the project, then type the following command.<br/>
```$ django-admin startproject WEB_APP```<br/>
### 2. Create a Django App
Now, we create a Django app. Each application written in Django consists of a Python package, that follows a certain convention. Django comes with a utility, that automatically generates the basic directory structure of the app, enabling us to focus on writing code rather than creating directories.<br/>
```$ cd WEB_APP```<br/>
```$ python manage.py startapp predict```<br/>
### 3. Editing “Django” apps.py
### 4. Editing views.py
### 5. Editing urls.py
### 6. Migrations and Superuser
The next step is to make migrations and create a superuser. Migrations are Django’s way of propagating changes we make to our models (adding a field, deleting a model, etc.) into our database schema. They’re designed to be mostly automatic, but we’ll need to know when to make migrations, when to run them, and the common problems we might run into. Superuser is a user who can login to the admin site.<br/>
#### Create migrations,<br/>
```$ python manage.py makemigrations```<br/>
```$ python manage.py migrate```<br/>
-->```makemigrations``` is responsible for creating new migrations based on the changes we make to our models.<br/>
-->```migrate``` is responsible for applying and unapplying migrations.<br/>
#### Create Superuser,
```$ python manage.py createsuperuser```<br/>
#### Enter the desired username and press enter.
```Username: user```
#### Next, enter desired email address:
```Email address: user@example.com```
#### The final step is to enter a password.
```Password: **********```<br/>
```Password (again): *********```<br/>
```Superuser created successfully.```<br/>
### 7. Run Server
Now, we are all set to deploy our Machine Learning model on the local host. Run server by using the command,<br/>
```$ python manage.py runserver```
### 8. Testing the API
Let’s test our REST API! We open our browser and type the following URL-```localhost:8002```
# Final Testing Video of COVID-19:
[![WEB-GUI](https://github.com/arjunparmar/COVID-19/blob/master/Data/Images/ScreenStart.png)](https://youtu.be/HM-Li8rkPOE)
# Website: [COVID19 DETECTION USING CHEST X-RAY](https://covid-19-detect-1327.herokuapp.com/)
