import pytest
from score import score
import pickle
import requests
import time
import subprocess
import warnings

warnings.filterwarnings("ignore")

vectorizer, clf = pickle.load(open("./bestModel.pkl", "rb"))
model = (vectorizer, clf)

def test_smoke():
    """Smoke test - check if the function runs without crashing"""
    prediction, _ = score("This is a test", model, 0.5)
    assert isinstance(prediction, bool)

def test_format():
    """Format test - check input/output formats and types"""
    prediction, propensity = score("This is a test", model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)

def test_prediction_values():
    """Test that prediction is binary (0 or 1 when converted to int)"""
    prediction, _ = score("This is a test", model, 0.5)
    assert prediction in [True, False]

def test_propensity_range():
    """Test that propensity is between 0 and 1"""
    _, propensity = score("This is a test", model, 0.5)
    assert 0 <= propensity <= 1

def test_threshold_zero():
    """Test that threshold 0 always gives prediction 1"""
    prediction, _ = score("This is a test", model, 0.0)
    assert prediction is True

def test_threshold_one():
    """Test that threshold 1 always gives prediction 0"""
    prediction, _ = score("This is a test", model, 1.0)
    assert prediction is False

def test_obvious_spam():
    """Test that obvious spam text gives prediction 1"""
    obvious_spam = "URGENT: Buy now! Free Viagra, Casino, Make money fast!!!"
    prediction, _ = score(obvious_spam, model, 0.5)
    assert prediction is True

def test_obvious_ham():
    """Test that obvious non-spam text gives prediction 0"""
    obvious_ham = "Hi John Doe, how are you doing?"
    prediction, _ = score(obvious_ham, model, 0.5)
    assert prediction is False

def test_flask():
    process = subprocess.Popen(["python", "app.py"])
    time.sleep(3)  # give the server time to start

    # Use form data instead of JSON
    form_data = {
        "text": "URGENT: Buy now! Free Viagra, Casino, Make money fast!!!",
        "threshold": 0.5
    }

    response = requests.post("http://127.0.0.1:5000/", data=form_data)

    assert response.status_code == 200
    html = response.text

    assert "Classification Result" in html
    assert "Propensity" in html
    assert "Prediction" in html

    process.terminate()
    process.wait()
    
'''
print(test_smoke())
print(test_format())
print(test_obvious_ham())
print(test_obvious_spam())
print(test_prediction_values())
print(test_propensity_range())
print(test_threshold_one())
print(test_threshold_zero())
'''

#test_flask()