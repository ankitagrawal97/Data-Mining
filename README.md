# Data-Mining
An Image Captioning project for data mining course 
Name: Ankit Agrawal, Ayush Sharma, Bishesh Bikram Shah & Nandan kumar
Enrollment Number: 171210010,171210016,171210060,171210040 (respectively)
Year: 3rd Year Computer Science and Engineering


# Dependencies

* Keras 1.2.2
* Tensorflow 0.12.1
* tqdm
* numpy
* pandas
* matplotlib
* pickle
* PIL
* glob


# How to run

0. Download Flickr8K Dataset from https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b
1. Extract Both Flickr8k zip files in this directory
2. Execute run.sh by  ./run.sh
3. It will create virtual environment, then install dependencies and finally launch jupyter notebook.
4. Select kernel named 'venv' and execute the notebook cell by cell and save weights.
5. Exit notebook and virtual environment will be deactivated automatically.

Note:
* 'Image Captioning InceptionV3.ipynb' is the code file containing python code.
* checkpoint weights are not uploaded (>25mb)
* The file containing default weights for InceptionV3 is not uploaded (>25mb).
* Images are saved in images folder

# Overview

> Using Flickr8k dataset since the size is 1GB. MS-COCO is 14GB!
> Used Keras with Tensorflow backend for the code. InceptionV3 is used for extracting the features.
  Since it has very less parameters as compared to VGG-16
> Using Beam search with k=3, 5, 7 and an Argmax search for predicting the captions of the images.
> The loss value of 3.1449 has been achieved.
> Everything is implemented in the Jupyter notebook which will hopefully make it easier to understand the code.
