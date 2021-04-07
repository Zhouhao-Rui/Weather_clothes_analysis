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

![][accuracy]

[accuracy]: ./img/accuracy.png