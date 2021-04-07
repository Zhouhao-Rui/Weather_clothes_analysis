# weather clothes data analysis

## 1. model selection and data set

The data set includes weather condition, temperature, wind speed and user preference.

- temperature: Expressed in degrees Celsius.

- weather condition
  - 0: sunny day
  - 1: cloudy day
  - 2: rainy day
- wind speed: Use kilometers/hour as the unit.
- preference1: user clothes preference
- preference2: user trousers preference

The model based on Keras framework, and I use one full-connected layer with relu activation function and one full-connected layer with softmax function.

After training, the accuracy is around 90%.

- male model

![][accuracy]

- female model

  ![][accuracy_female]

## 2. load the model and predict in the api

- how to build the api
  - with the flask framework
- where to deploy
  - deploy on my own cloud server(centos7)
  - with the uwsgi & nginx

[accuracy]: ./img/accuracy.png
[accuracy_female]: ./img/accuracy_for_female.png

