# Practical machine learning for biosignal analysis

## Course description 
This graduate level course will introduce practical machine learning concepts and tools, and will exemplify their application to the analysis of biological signals and images, including EEG, ECog, brain imaging, electrophysiology, and image recognition.


## Prerequisites 
1. Basic Python programming
2. Calculus
3. Linear algebra (suggested)


## Instructor

**Andrea Giovannucci**
Assistant Professor, Biomedical Engineering Department
**email**: agiovann@email.unc.edu
**office**: Mary Ellen jones 9202B 
**office hours**: Tue-Th, 4.45 PM - 5.45 PM 

## Textbook

*Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems 2nd Edition* (**available in October 2019**). [Here](https://www.acm.org/membership/offers/skill-up-oreilly) a cheap way to get access to this book and other 50000. It can be accessed through the Safari O'Really learning platform [O'Reilly](https://learning.oreilly.com/) which comes fo free with ACM membership ($75).

> Past Edition. Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. Aurelien Geron. Ed. O' Reilly.

## Resources

1. App to answer questions during the lecture. [Poll everywhere](https://poll.unc.edu/)
2. [Course repository](https://github.com/agiovann/machine-learning-BMME890/)
3. [Sakai](https://sakai.unc.edu) 
4. [Slack](https://bmme890.slack.com)
5. [Anaconda](https://www.anaconda.com/distribution/) Python distribution
6. Prepare slides in Jupyter Notebooks with [RISE](https://github.com/damianavila/RISE)
7. [scikit-learn](https://scikit-learn.org)
8. [kaggle](https://www.kaggle.com/competitions) competitions

It is **strongly advised** to work with a Linux or MAC OS machine (you can make it work on windows with some customization). You have two possibilities:

* If you have a window machine, you can install linux using a virtual machine (such as [parallels](https://www.parallels.com/), [vmware](https://www.vmware.com/products/workstation-pro/workstation-pro-evaluation.html) or others)
* You can use services like
    * Microsoft Azure
    * Amazon Web Services
    * IBM Cloud
    * VMware
    * Google Cloud Platform
    * Digital Ocean    

## Learning objectives 
By the end of the course students will be able to:

1. use Github to share, contribute and extend open-source code
2. explore, visualize and interpret datasets using the matplotlib and pandas Python libraries
3. identify and use properly machine learning terms and concepts: supervised and unsupervised learning, variance-bias tradeoff, underfitting and overfitting, batch and online learning, testing and validating, classification and regression
4. formalize a problem in machine learning terms, solve with the appropriate method in the scikit and keras learning packages, and visualize solution metrics
	* assimilate principles and limitations of the following machine learning techniques: classification, regularized linear regression, dimensionality reduction, clustering, kernel methods, neural networks and deep learning   
	* set up a pipeline for image classification
	* integrate machine learning algorithms in biosignal analysis pipelines: analyze EMG, EEG, electrophysiological, and imaging data


 
## Homework & Project (total 200 points)

* **HW1**. Setting up (20 pts)
* **HW2a**. Getting started with Jupyter Notebooks  and visualization(5 pts)
* **HW2b**. Recap algebra and statistics (15 pts) 
* **HW3**. Course project description and presentation (20  pts)
* **HW4**. *Kaggle Competition*. Regression Problem (20 pts) 
* **HW5**. *Kaggle Competition*. Classification Problem (20 pts)
* **Optional. Up to 20 points**
    *prepare a 30 minutes lecture to explain to the class an argument (one can choose among *support vector machines, clustering, gaussian mixtures, PCA and ICA*). One needs at least 85 points by week 7. 
* **HW6**. Unsupervised Learning and Clustering. Spike sorting (20 pts)
* **HW7**. Convolutional Networks (20 pts)
* **HW8**. Course project presentation, 15 minutes. (60 pts)**

## Class quizzes (total 50 points)

* There will be 3-7 during classes, will cover the material that is explained and one question regarding the material covered in the previous class, we will use [Poll Everywhere](poll.unc.edu)
* Max points per class is 2

## Evaluation (see points above)

The final grade will be computed as follows

* 20% classroom quizzes 
* 48% homework assignments
* 32% project

## Course schedule 

| Week |   In class  | Readings  | Deliverables (due) |
| :------------- 	| ------------- |:-------------:| -----:|
| #1 		Intro	| overview/anaconda/poll/github | slides | -- |
| #2 		Recap	| spyder/jupyter/pandas/visualization |slides|HW1 (Th)|
| #3 		Recap	| linear algebra/statistics | slides |  |
| #4 		Recap	| ML overview/kaggle |chapter 1-2| HW2 (Th)|
| #5 		HW		| project presentations | -- |  HW3 (Tue) |
| #6 		SL| classification| chapter 3 | -- |
| #7 		SL| regression| chapter 4 | --   |
| #8 		SL| support vector machines | chapter 5 | HW4 |
| #9 		UL| dimensionality reduction | chapter 8 | HW5 |
| #10 		UL| clustering/gaussian mixtures/spike sorting | chapter 9 | -- |
| #11 		DL	| setting up/intro to ANNs  | chapter 10-13| HW6 |
| #12 		DL	| intro to ANNs | chapter 10-13 |--|
| #13 		DL	| convolutional nets | chapter 14 |--|
| #14 		DL	| recurrent deep nets | chapter 15  |HW7|
| #15 		HW	| project presentations | --  | Project |
| #16 		Extra	| EEG, ECoG | slides  | Project |
| #17 		Extra	| Brain Imaging | slides  | Project |

legend: SL (Supervised Learning), UL (Unsupervised Learning), DL (Deep Learning), HW (Homework)


## PROGRAM 
1. Overview of the course plus method of evaluation
	* tools set up
	* course outline
2. Background
	* Getting familiar with GitHub 
	* Using Pandas and visualization
	* Linear algebra/statistics recap  
3. Introduction to Machine learning
	* Discipline overview
	* A practical Example
4. Supervised Learning
	* Classification
	* Regression
	* Support Vector Machines
5. Unsupervised Learning
	* Dimensionality Reduction
	* Clustering
	* Gaussian Mixtures
6. Deep Learning
	* Introduction
	* Multilayer Perceptron
	* Back-propagation and training networks
	* Convolutional Networks
	* Recurrent Networks
7. Biomedical applications
	* EEG, ECog
	* brain imaging

## Datasets

* EEG/EcOG/
	* install with ```conda install -c conda-forge mne```
	* [mne website](https://martinos.org/mne/stable/index.html) 
* [spike sorting](https://github.com/flatironinstitute/mountainsort_examples)

## ACCESSIBILITY RESOURCES

The University of North Carolina at Chapel Hill facilitates the implementation of reasonable accommodations, including resources and services, for students with disabilities, chronic medical conditions, a temporary disability or pregnancy complications resulting in difficulties with accessing learning opportunities.

All accommodations are coordinated through the Accessibility Resources and Service Office. See the ARS Website for contact information: https://ars.unc.edu or email ars@unc.edu.

Relevant policy documents as they relation to registration and accommodations determinations and the student registration form are available on the ARS website under the About ARS tab.	
	

