# Sentiment-Analysis-NLP
Trained a Stochastic Gradient Descent Classifier with different ngram model representations with Tf-idf, to classify imdb reviews as being positive or negative.

Dataset used for this project: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

### The Dataset
The dataset contains 50,000 reviews from IMDB. Number of positive and negative reviews are 25k each. Negative reviews have scores ≤ 4 out of 10 while a positive review ≥ 7 out of 10.

There’s a `train/` and `test/` folder, each with `pos/` and `neg/` directories containing text files of reviews. The reviews in the `pos/` and `neg/` folders are combined to create a training and testing file. 

This data gets extracted and pre-processed to feed into the n-gram classifier.

### The Classifier
Stochastic gradient descent classifier requires an array X of shape `(n_samples, n_features)` holding the training samples, and an array y of shape `(n_samples,)` holding the target values (class labels) for the training samples.

Numerical representation of text data is achieved using Bag-of-Word models or n-grams. sci-kit learn vectorizers do just that, in particular, the `CountVectorizer` and `TfidfVectorizer`.

The vectorizer can be modified to remove stopwords, the n-gram range and idf re-weighting (only in the case of `TfidfVectorizer`).

`fit_transform` on the training data learns the vocabulary dictionary and returns document-term matrix - stored as `X_train`. `transform` transforms the test data to the document-term matrix - stored as `X_test`.

The `fit` method on `X_train` and `y_train` fits a linear model with Stochastic Gradient Descent. The `predict` method is then used to predict class labels for samples in `X_test`.
