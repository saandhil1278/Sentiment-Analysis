# Sentiment Analysis using NLTK & Transformers

## [1. Introduction](#introduction)
The dataset used in this project is sourced from Kaggle, specifically the Amazon Fine Food Reviews dataset. This project employs various approaches for sentiment analysis:

### [1.1 Traditional Approach using Python NLTK Toolkit](#traditional-approach-using-python-nltk-toolkit)
Initially, we used the traditional approach with the Python NLTK toolkit to fit the model for sentiment analysis.

### [1.2 VADER Model: Valence Aware Dictionary and Sentiment Reasoner](#vader-model-valence-aware-dictionary-and-sentiment-reasoner)
We then used the VADER model, which utilizes a "bag of words" approach to perform sentiment analysis.

### [1.3 RoBERTa Model with Hugging Face](#roberta-model-with-hugging-face)
Next, we employed the RoBERTa model provided by Hugging Face, leveraging transformers and the method of transfer learning. Although initially trained on Twitter reviews, the RoBERTa model can be fine-tuned on the Amazon Fine Food Reviews dataset.

### [1.4 Comparison between VADER and RoBERTa Model](#comparison-between-vader-and-roberta-model)
A comparison was made between the VADER model and the RoBERTa model to evaluate their effectiveness.

### [1.5 Pretrained Pipelines for Sentiment Analysis Using Hugging Face](#pretrained-pipelines-for-sentiment-analysis-using-hugging-face)
We explored pretrained pipelines for sentiment analysis using the Hugging Face library.

**Libraries Used:** numpy, pandas, seaborn, matplotlib, nltk, tensorflow, keras, transformers

## [2. Model Preprocessing](#model-preprocessing)

### [2.1 Quick EDA](#quick-eda)
Performed exploratory data analysis (EDA) using various visualizations such as bar plots, dist plots, and count plots.

### [2.2 Basic NLTK Tokenization](#basic-nltk-tokenization)
Applied NLTK's word_tokenize method for tokenization, followed by part-of-speech (POS) tagging to identify the grammatical tags of words such as determiners, nouns, pronouns, and verbs.

## [3. Model Evaluation](#model-evaluation)

### [3.1 VADER Sentiment Scoring](#vader-sentiment-scoring)
Used NLTK's SentimentIntensityAnalyzer to obtain negative, neutral, and positive scores of the text. The VADER model follows a "bag of words" approach where:
- Stop words are removed as they do not contribute to sentiment.
- Each word is scored individually, and the total score is computed.
- Created an object of SentimentIntensityAnalyzer, fitted data, and conducted sentiment analysis using polarity scores. We considered 5-star ratings as positive reviews and vice versa.

### [3.2 Drawbacks of VADER](#drawbacks-of-vader)
The VADER model scores each word individually but cannot comprehend the context and relationships among words, making it less effective for sentences with sarcasm, slang, idioms, etc. This limitation is addressed by the RoBERTa model, which:
- Uses a model trained on a large corpus of data.
- Accounts for the context related to other words.

We performed transfer learning with the RoBERTa model from Hugging Face on the Amazon Fine Food Reviews dataset.

### [3.3 Combining Scores from VADER and RoBERTa](#combining-scores-from-vader-and-roberta)
Combined the negative, neutral, and positive scores from both the VADER and RoBERTa models for each sentence in a table and concatenated these columns into the original dataframe.

### [3.4 Using Hugging Face Pipelines](#using-hugging-face-pipelines)
A quick and easy way to run sentiment predictions is using Hugging Face pipelines:
```python
from transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis")
example = sent_pipeline("Make sure to be happy and enjoy life")
# Output: [{'label': 'POSITIVE', 'score': 0.9923}]

```
4. Conclusion
The implementation of sentiment analysis on the Amazon Fine Food Reviews dataset showcases the practical application of both traditional and advanced machine learning models. The traditional NLTK approach and VADER model provide a foundational understanding of sentiment analysis. However, due to the limitations of these models in understanding context and complex language constructs, advanced transformer models like RoBERTa are more effective. The transfer learning capabilities of RoBERTa, combined with the ease of use provided by Hugging Face pipelines, demonstrate significant improvements in sentiment analysis accuracy and contextual understanding. This project highlights the importance of choosing the right model and techniques based on the complexity and nature of the dataset.
