import pickle

from flask import Flask, request, render_template
import pandas as pd
from matplotlib import pyplot as plt

# Declare a Flask app
app = Flask(__name__)

# Training dataset
train = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Twitter-Sentiment-Analysis/master/train_tweet.csv")
# Test dataset
test = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Twitter-Sentiment-Analysis/master/test_tweets.csv")

print(train.head(), '\n- - - - - - - - - - - - - - - - - -')
# non-racist/sexist related tweets
print('Non-racist/Sexist related tweets: ', sum(train["label"] == 0))

# racist/sexist related tweets
print('Racist/Sexist related tweets: ', sum(train["label"] == 1))
print('- - - - - - - - - - - - - - - - - -')

# Remove special characters using the regex library
import re

# set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile(
    "(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

import preprocessor as p


# custom function to clean the dataset (combining tweet_preprocessor and regular expression)
def clean_tweets(df):
    tempArr = []
    for line in df:
        # send to tweet_processor
        tmpL = p.clean(line)
        # remove punctuation
        tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower())  # convert all tweets to lower cases
        tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
        tempArr.append(tmpL)
    return tempArr


train_tweet = clean_tweets(train["tweet"])
train_tweet = pd.DataFrame(train_tweet)

# Append cleaned tweets to the training data
train["clean_text"] = train_tweet

from sklearn.model_selection import train_test_split

# extract the labels from the train data
y = train.label.values

# use 90% for the training and 10% for the test, Because of the existence of test dataset
x_train, x_test, y_train, y_test = train_test_split(train.clean_text.values, y,
                                                    stratify=y,
                                                    random_state=1,
                                                    test_size=0.1, shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)

from sklearn import svm

# classify using support vector classifier
svm = svm.SVC(kernel='linear', probability=True)

# fit the SVC model based on the given training data
prob = svm.fit(x_train_vec, y_train).predict_proba(x_test_vec)

# perform classification and prediction on samples in x_test
y_pred_svm = svm.predict(x_test_vec)

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy score for SVC is: ", accuracy_score(y_test, y_pred_svm) * 100, '%')

print("Classification Report for SVC is: \n", classification_report(y_test, y_pred_svm),
      "- - - - - - - - - - - - - - - - - - -")

from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_svm)
print('Confusion Matrix: \n', cm)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

pickle.dump(svm, open('svm_model.pkl', 'wb'))


@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    global outp
    if request.method == "POST":

        # Unpickle classifier
        clf = pickle.load(open('svm_model.pkl', 'rb'))

        # Get values through input bars
        tweet = request.form.get("tweet")

        vec = vectorizer.transform([tweet])

        # Get prediction
        prediction = clf.predict(vec)

        global colr
        if str(prediction) == '[0]':
            global outp

            outp = 'Not contain Hatred text'
            colr = 'darkgreen'
        else:
            outp = 'It contain Hatred text'
            colr = '#cd0000'

    else:
        outp = ""
        colr = ""

    return render_template("index.html", Output=outp, colr=colr)


# Running the app
if __name__ == '__main__':
    app.run(debug=True)
