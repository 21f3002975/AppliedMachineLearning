import os
import spacy
import pickle
import warnings


warnings.filterwarnings("ignore")

# Load spaCy model and stop words once at the module level
try:
    nlp = spacy.load('en_core_web_sm')
    stop_words = nlp.Defaults.stop_words
except Exception as e:
    raise RuntimeError(f"Error loading spaCy model: {str(e)}")

def tokenize_text_input(text):
    # Tokenize the given text using spaCy, removing stop words and non-alphabetic tokens.

    try:
        doc = nlp(text.lower())
        tokens = [token.text for token in doc if token.is_alpha and token.text not in stop_words]
        return tokens
    except Exception as e:
        raise RuntimeError(f"Error in tokenization: {str(e)}")

def score(text, model, threshold = 0.5):
    if not isinstance(text, str):
        raise TypeError("Expected 'text' to be string.")
    
    vectorizer, clf = model

    if not hasattr(clf, 'predict_proba') or not callable(clf.predict_proba):
        raise TypeError("Model must have a callable 'predict_proba' method.")
    
    if not isinstance(threshold, float):
        raise TypeError("Threshold value should be a float.")
    
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold has to be a value between 0.0 and 1.0.")
    
    tokens = tokenize_text_input(text)
    input = [" ".join(tokens)]
    vectorized_input = vectorizer.transform(input)
    
    propensity = clf.predict_proba(vectorized_input)[0][1]
    prediction = propensity >= threshold

    return bool(prediction), round(propensity,3)

