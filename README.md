

# CaptchaResolver
Student project from subject named Soft computing. Solving problem of different types of CAPTCHA patterns on web forms.

Application consists of 2 parts : 
* Detection of CAPTCHA patterns on web forms
* Solving found CAPTCHAs.

Web forms contains one or more CAPTCHA patterns. CAPTCHA patterns contains various number of characters. 
Faster R-CNN is used for detecting CAPTCHA patterns. For solving detected CAPTCHAs is used RNN (LSTM). There is also example of RNN GRU for solving CAPTCHAs, but LSTM gives better results(6%-7% accuracy difference). 

## Preview


### CAPTCHA Detection

At first, application tries to find all regions with CAPTCHA pattern on given web form. We used Faster R-CNN for detection of these regions (results are shown with drawn boxes on pictures). 

![detection 1](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/5.png)
![detection 2](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/4.png)
![detection 3](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/6.png)

RPN is trained with 10 epochs and each epoch contained 500 pictures with augmentation.
Results:
* Classifier accuracy for bounding boxes from RPN: 0.95

* Loss RPN classifier: 0.02459541053066148

* Loss RPN regression: 0.014727079278265592

* Loss Detector classifier: 0.13960414096550083

* Loss Detector regression: 0.06957839820068329

* Total loss: 0.2485050289751112

![loss_rpn_cls-regr](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/diagrams/Loss_rpn_cls-regr.png)
![accuracy_overlapping_bboxes](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/diagrams/accuracy-overlapping_bboxes.png)
![loss_class_cls-regr](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/diagrams/loss_class_cls-reg.png)
![total_loss-elapsed_time](https://github.com/acastef/CaptchaResolver/blob/master/images/detection/diagrams/total_loss-elapsed_time.png)


### CAPTCHA Solver

Before passing detected CAPTCHAs to RNN, necessary images preprocessing had performed. Preprocessing included image thresholding and morphological transformations.

![preprocessing_images](https://github.com/acastef/CaptchaResolver/blob/master/images/processor/images.png)

We used LSTM for solving CAPTCHAs which had 82.779% accuracy, instead of GRU which firstly was implemented, and scored 76.6% accuracy.

## Requirements

* Anaconda for Python 3.x
* If you have problems with OpenCV, TensorFlow or Keras libraries, use pip to install them.

## Usage guide

You need to download train weights for CAPTCHA detection network. Weights are available on this link, which will be available until 26.2.2019. https://drive.google.com/file/d/1nFB4hp_-rXrBKDlqbXH6R_5KK593Afka/view?fbclid=IwAR3aAQz7DuQ0-XgclSPHkZNcQFkWt7kUkMZvFfG2oVeioVVmc2aWeev9npI

You need to put this file in next folder in project: CaptchaResolver/model/faster_rcnn.
File should be named model_frcnn_vgg.hdf5.
To run demo, you need to follow this next commands: 
* `cd src/detection` # make your current working directory src/detection
* `python demo.py` #python command should be anaconda python interpreter

To run Faster R-CNN separately from RNN on test data, follow next commands:
* `cd src/detection` # make your current working directory src/detection
* `python test.py` 

This action should last at least one hour. You can skip this command, and results of this command are placed in [this](https://github.com/acastef/CaptchaResolver/blob/master/test_logs/detection.log) file.

To run RNN LSTM separately from Faster R-CNN on test data, follow next commands:
* `cd/src/rnn`
* `python lstm_predict.py`

If you want to skip this commands, results are placed in [this](https://github.com/acastef/CaptchaResolver/blob/master/test_logs/lstm.log) file. 

To run RNN GRU separately from Faster R-CNN on test data, follow next commands:
* `cd/src/rnn`
* `python gru_predict.py`

If you want to skip this commands, results are placed in [this](https://github.com/acastef/CaptchaResolver/blob/master/test_logs/gru.log) file.
