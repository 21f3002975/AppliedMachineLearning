from flask import Flask, request, render_template, jsonify
import pickle
from score import score
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


vectorizer, clf = pickle.load(open("./bestModel.pkl", "rb"))
model = (vectorizer, clf)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        threshold = float(request.form['threshold'])
        prediction, probability = score(text, model, threshold)

        return render_template('result.html', prediction=prediction, propensity=probability)
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)