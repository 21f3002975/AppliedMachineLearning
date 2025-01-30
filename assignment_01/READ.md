### Problem Statement

#### [16 Jan 2025] Assignment 1: Prototype (due 30 Jan 2025)

Build a prototype for SMS spam classification:

1. In `prepare.ipynb`, write the functions to:
   
   - Load the data from a given file path
   - Preprocess the data (if needed)
   - Split the data into train/validation/test
   - Store the splits at `train.csv`, `validation.csv`, and `test.csv`

2. In `train.ipynb`, write the functions to:
   
   - Fit a model on train data
   - Score a model on given data
   - Evaluate the model predictions

3. Validate the model:
   
   - Fit on train
   - Score on train and validation
   - Evaluate on train and validation

4. Fine-tune hyper-parameters using train and validation (if necessary)

5. Score three benchmark models on test data and select the best one.

---

### Notes:

- You may download SMS spam data from [UCI dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).
- You may refer to [this Python data science resource](https://radimrehurek.com/datascience_python/) for building a prototype.
- You may refer to the first three chapters of [Stat Learning](https://www.statlearning.com/) for basic ML concepts.
- You may refer to the Solution Design example covered in class as a guideline for experiment design.

---

### To install Spacy English Language Model

```bash
python -m spacy download en_core_web_sm