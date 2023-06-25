from flask import Flask, render_template, session, url_for, redirect
import numpy as np
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from tensorflow.keras.models import load_model
import joblib
import yfinance as yf


def return_predictions(model, scaler, json_content):  
  stock_quote = yf.download(json_content['stock'])
  new_df = stock_quote.filter(['Close'])
  last_60_days = new_df[-60:].values
  last_60_days_scaled = scaler.transform(last_60_days)
  X_test = []
  X_test.append(last_60_days_scaled)
  X_test = np.array(X_test)
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  pred_price = model.predict(X_test)
  pred_price = scaler.inverse_transform(pred_price)
  return str(pred_price[0][0])



app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class StockForm(FlaskForm):
  stock = StringField('Stock')
  submit = SubmitField('Predict tomorrow\'s closing price')

@app.route("/", methods=['GET', 'POST'])
def index():
  form = StockForm()
  if form.validate_on_submit():
    session['stock'] = form.stock.data
    return redirect(url_for('prediction'))
  return render_template('home.html', form=form)

model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/prediction')
def prediction():
  content = {}
  content['stock'] = session['stock']
  results = return_predictions(model, scaler, content)
  return render_template('prediction.html', results=results)

if __name__ == '__main__':
  app.run()