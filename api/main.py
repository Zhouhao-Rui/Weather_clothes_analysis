from flask import Flask, request, jsonify
from keras.models import load_model

# load models
male_model = load_model('male.h5')

# labels
label_dict = {
  0: 'coat and jeans',
  1: 'coat and long',
  2: 'hoodie and jeans',
  3: 'hoodie and long',
  4: 'hoodie and short',
  5: 'jacket and jeans',
  6: 'jacket and long',
  7: 'shirt and jeans',
  8: 'shirt and long',
  9: 'shirt and short'
}

app = Flask(__name__)

@app.route("/")
def hello():
  return "This is the weather_analisis API"

@app.route("/male")
def male_analysis():
  # get the data from the app
  temp = int(float(request.args.get("temp")))
  weather = int(float(request.args.get("weather")))
  wind = int(float(request.args.get("wind")))
  prefer1 = int(float(request.args.get("prefer1")))
  prefer2 = int(float(request.args.get("prefer2")))

  # predict the result
  input_data = [[temp, weather, wind, prefer1, prefer2]]
  result = male_model.predict(input_data, batch_size=None, verbose=0, steps=None)
  result_list = result[0].tolist()
  label_index = result_list.index(max(result_list))
  label_val = label_dict[label_index]
  return label_val

if __name__ == '__main__':
  app.run(debug=True)

