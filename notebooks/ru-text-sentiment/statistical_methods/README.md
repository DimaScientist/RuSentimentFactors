# Text sentiment statistical analysis


## Directory description

Sentiment statistical analysis for:
* [unigrams](sentiment_analysis_1_gram.ipynb)
* [bigrams](sentiment_analysis_2_gram.ipynb)
* [trigrams](sentiment_analysis_3_gram.ipynb)
* [clustering](clustering.ipynb)


## Results

Accuracy by F1-score

| Model \ N-gramm              | Unigrams | Bigrams | Trigrams |
|------------------------------|----------|---------|----------|
| Naive Bayes                  | 0.63     | 0.49    | 0.42     |
| SVM                          | 0.58     | 0.42    | 0.41     |
| LinearSVM                    | 0.62     | 0.48    | 0.38     | 
| Logistic regression          | 0.62     | 0.47    | 0.37     |
| КNN                          | 0.5      | 0.43    | 0.36     |
| Decision tree                | 0.58     | 0.43    | 0.37     |
| Gradient boosting (Catboost) | 0.41     | 0.51    | 0.4      |