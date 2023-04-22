# Tweet Sentiment Analysis

School of Computer Science and Computer Engineering \
Nanyang Technological University \
Lab: A127 \
Team: 2

## Our Team
| Name | Parts Done | Github ID |
|---|:---:|---|
| Tan Hee | Exploratory Data Analysis (EDA), Data Preprocessing, VADER model, Github Repo and Report | @tanhee2000 |
| Kerwin Soon | Exploratory Data Analysis (EDA), Data Preprocessing, Naive Bays Classifier | @kerwinsoon |
| Lim Zheng Guang | Support Vector Classifier, Recurrent Neural Network | @fixthelights |

## About

This is our mini project for our module **SC1015 - Introduction to Data Science and Artificial Intelligence**.

## Our Motivation 

We want to gain insights into people's daily lives by analyzing their tweets on the platform - whether they are being positive, negative or neutral based on the text they are tweeting.

## Dataset

The [dataset](https://www.kaggle.com/code/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model/notebook) we used was taken from Kaggle, in the `Tweet Sentiment Extraction` competition. The goal of the competiton is actually **not to predict** the sentiments, but to **identify the specific word or phrase** that affects the sentiment. The competition was said to be unique by one of the participants.

Included in the dataset are 4 columns:
- textId `unique ID for each piece of text`
- text `the text of the tweet`
- selected text `the general sentiment of the tweet`
- sentiment `the text that supports the tweet's sentiment`

## Problem Statement

To accurately predict and classify the sentiment of the tweet as positive, negative or neutral using different models.

## Exploratory Data Analysis 

  ### Seaborn heatmap
  
  **3 types of sentiment:**
  1. Neutral (Count: 11117)
  2. Negative (Count: 7781)
  3. Positive (Count: 8582)
  
  <img src="https://user-images.githubusercontent.com/90170330/233783250-ad4e47ab-d6b5-4699-8a7f-160618157717.png" width="600">
  
  ### Seaborn count plot
  
  <img src="https://user-images.githubusercontent.com/90170330/233783158-2d52a6d0-5a76-402c-b369-758f33f7b811.png" width="600">  
  
  
  ### Seaborn histogram plot
  
  - All 3 sentiments have similar distribution of words in a tweet
  - Average distribution is 12 words in a tweet
  
  <img src="https://user-images.githubusercontent.com/90170330/233781878-eac9a9df-c05b-4647-8ae8-fbe78bac25e6.png" width="600">
  
  ### Treepmap Charts
  
  #### Most common positive words
  
  ![treemap positive](https://user-images.githubusercontent.com/90170330/233781919-696285e3-35bb-4a48-a2a9-f31684a3539d.jpg)
  
  - The top 3 most common positive words in a tweet are `day`, `good` and `love`.
  - `day` is 1179 of the total positive words
  - `good` is 972 of the total positive words
  - `love` is 872 of the total positive words
  
  #### Most common neutral words
  
  ![treemap neutral](https://user-images.githubusercontent.com/90170330/233782728-53423ea4-28eb-4711-9468-06d33fa77d9b.jpg)
  
  - The top 3 most common neutral words in a tweet are `get`, `go` and `http`.
  - `get` is 626 of the total neutral words
  - `go` is 574 of the total neutral words
  - `http` is 526 of the total neutral words
  
  #### Most common negative words
    
  ![treepmap neg](https://user-images.githubusercontent.com/90170330/233782821-7a171533-50e6-4314-aba8-66e2bc87146c.jpg)
  
  - The top 3 most common negative words in a tweet are `like`, `get` and `miss`.
  - `like` is 478 of the total negative words
  - `get` is 435 of the total negative words
  - `miss` is 421 of the total negative words
 
  ### Further Analysis of Unique Patterns
  1. Tweets were taken around `mother's day` (351 rows contain `mother`)
      - This may affect our model training, creating biases around the word `mother`.
      
      ![mother'sday](https://user-images.githubusercontent.com/90170330/233784133-cc5d9b73-b7b5-428e-98e6-2a2a28d535ea.png)

  2. There are a lot of `URLs` in the text (1223 rows contain `URLs`)
  
      ![http](https://user-images.githubusercontent.com/90170330/233784141-e6449c41-e97d-4609-924f-1c9292a428ff.png)

  3. There were a lot of `censored words` replaced with `****` in the dataset (1000 rows contain `****`) 
  
      ![asterisk](https://user-images.githubusercontent.com/90170330/233784147-2b96a842-b1f2-4821-9f24-2d270a490fba.png)

  ### Preprocessing Methods

  We'll explore some `Data Preprocessing` techniques including the following:
  | Data Preprocessing Techniques | Definition |
  |---|---|
  | Tokenizing | Breaking a text based on the token (a meaningful unit of text) | 
  | Filtering Stop Words | Filter words you want to ignore of your text when processing it | 
  | Stemming | A text processing task in which you reduce words to their root, which is the core part of a word. | 
  | Lemmatizing | Reduce words to their core meaning, but it will give you a complete English word that make sense on its own instead of just a fragment of a word like 'discoveri'. | 
  | Tagging parts of speech | Part of speech is a grammatical term that deals with the roles words play when you use them together in sentences. | 
  | Chunking | While tokenizing allows you to identify words and sentences, chunking allows you to identify phrases. | 
  
  ### Data Preprocessing
  
  1. Data Cleaning
      - Remove Hyperlinks
      - Remove Stopwords
      - Stemming of words using `PorterStemmer`
      
      ![data preprocessing techniques](https://user-images.githubusercontent.com/90170330/233783736-17754681-4088-413c-a499-949fab3eefd4.jpg)

  2. Feature Engineering
      - Added a `cleantext_word_count` column for use during model training later on.
  
## Model Training and Evaluation

We will be looking at 3 different models with different complexities to have a general idea of the performance to expect, and it'll make for good comparisions when we evaluate our models.

1. [Lexicon-Based Sentiment Analysis with VADER](#Lexicon-Based-Sentiment-Analysis-with-VADER)
2. scikit-learn Models
    1. [Naive Bays Classifier](#Naive-Bays-Classifer)
    2. [LinearSVC Classifier](#LinearSVC-Classifer)
3. [Recurrent Neural Network using Keras](#Recurrent-Neural-Network-using-Keras)

## Lexicon-Based Sentiment Analysis with VADER 

> `VADER` - **Valence Aware Dictionary and sEntiment Reasoner**

We'll be using a **`pretrained`** model to create a performance baseline. Do note that this model does not use Machine Learning, it works by giving each word in a sentence a score based on an internal `valence dictionary`, then aggregates the scores to provide a overall sentiment score for a sentence.

Since lexicon-based approaches are inherently `dumb` in that it simply refers to a dictionary for predictions, the performance of the predictor is expected to be lower than more advanced models. This provides us with a baseline performance benchmark, which can be used to evaluate our in-house models in the future.

**Confusion Matrix for VADER**
  
<img src="https://user-images.githubusercontent.com/90170330/233784489-9085951b-5b65-4908-82be-22339c92e4d5.jpg" width="600">

**F1 score for VADER**

-  We found an average `F1 Score` of `0.64` for VADER.

<img src="https://user-images.githubusercontent.com/90170330/233784656-6c404acc-4714-43a5-839b-e837b73fb519.png" width="600">  

## Naive Bays Classifer

The `Naïve Bayes` classifier is a supervised machine learning algorithm, which is used for classification tasks, like text classification. It is also part of a family of generative learning algorithms, meaning that it seeks to model the distribution of inputs of a given class or category.

It is known to perform well for text classification tasks such as sentiment analysis, and is very fast. It assumes that our predictors (features) are independent, and if that holds, the performance can be better than other Machine Learning models such as `logistic regression` or `decision trees`. However in practice, it is diffcult to find predictors that are independent, hence the model may not perform as well.

We use `sklearn's` `MultinomialNB`. Taken from their documentation:
> `MultinomialNB` implements the naive Bayes algorithm for multinomially distributed data, and is one of the two classic naive Bayes variants used in text classification (where the data are typically represented as word vector counts, although tf-idf vectors are also known to work well in practice). 

We split our data 80, 20 for training and testing respectively. Upon evaluation, the model performed decently, scoring an `F1 Score` of `0.65`.

**Confusion Matrix for Naive Bays**
  
<img src="https://user-images.githubusercontent.com/90170330/233784855-6c06f829-95ae-4705-b746-3777d7e33155.png" width="600">  


**F1 score for Naive Bays**

-  We found an average `F1 Score` of `0.65` for Naive Bays.

<img src="https://user-images.githubusercontent.com/90170330/233784862-651f43e5-c3d6-4281-bba3-d065433b6524.png" width="600">  

## LinearSVC Classifer

Support Vector Classifier, is a supervised machine learning algorithm typically used for classification tasks. SVC works by mapping data points to a high-dimensional space and then finding the optimal hyperplane that divides the data into two classes.

Compared to `Naive Bays Classifer`, the `Support Vector Classfier` is known to work as well or even better. Hence we put them side by side to see the performance we can get from `LinearSVC`.

For `Vectorizing`, we used the `TfidVectorizer` which "converts a collection of raw documents to a matrix of TF-IDF features". We found that this method of text `Vectorizing` achieves a better performance for our `Support Vector Classifer` as compared to the `CountVectorizer` used by the `Navie Bays Classifer`.

**Confusion Matrix for LinearSVC**
  
<img src="https://user-images.githubusercontent.com/90170330/233785152-800fbb3f-3c28-4e4f-9a94-8e2b26afc528.png" width="600">  

**F1 score for LinearSVC**

-  We found an average `F1 Score` of `0.67` for LinearSVC.

<img src="https://user-images.githubusercontent.com/90170330/233785172-079a43c8-5674-43a1-bdb1-772f27ad328f.png" width="600">  

## Recurrent Neural Network using Keras

Using a LTSM Neural Network allows us to achieve the best performance yet. 

Using Keras, we create layers in our Neural Network to fit our data.
> Keras is an open-source software library that provides a Python interface for artificial neural networks.

Layers are as follows:
1. Embedding Layer
2. LSTM layer
3. Dense layer

We create an Embedding layer to learn `word embeddings` from scratch. This converts our wordy sentences into `dense vectors`. These dense vectors allow our Neural Network models to converage easier compared to spares vectors.

Next we have LSTM layer, which is a type of RNN especially performant in text classification tasks. We define the number of hidden units within the layer, and the dropout rate. The dropout rate is used to randomly switch off neurons during training, creating regularization and reducing the tendency to `overfit` during training.
> `LTSM` - Long Short Term Memory. It stores a large amount of previous training data in memory. This is useful for text since the contexts of words in sentences depends on previous words before them.

Lastly, we use a Dense layer with 3 output units to represent our 3 classes: `negative, neutral, positive`. We use `softmax` as an activation function to collect the probabilistic values of each class, which is useful in classification.

**Evaluation**

After training, we evaluated the model and computed the `F1 Scores` and `Confusion Matrix`, to visualize the performance of our Neural Network. We found an average `F1 Score` of `0.72`, which is the best performer compared to all other models we evaluated.

**Confusion Matrix for LTSM Neural Network**
  
<img src="https://user-images.githubusercontent.com/90170330/233785325-a1677c8a-9a42-4e2c-9f33-8e39e0f26e8c.png" width="600">  

**F1 score for LTSM Neural Network**

-  We found an average `F1 Score` of `0.73` for LTSM Neural Network.

<img src="https://user-images.githubusercontent.com/90170330/233785342-c7a16344-7aaf-408d-835c-9b8ff8214b1d.png" width="600">  


## Bidirectional LTSM Model

Now we compare the LTSM model with one that has bidirectional layers, which may improve provide greater performance. The bidirectional layers help the RNN catch more complex patterns in text that may otherwise be missed by a normal RNN layer.

**Confusion Matrix for Bidirectional-LTSM Neural Network**
  
<img src="https://user-images.githubusercontent.com/90170330/233785466-36d19db0-cdf2-4d6f-b2a3-8b430917b218.png" width="600">  


**F1 score for Bidirectional-LTSM Neural Network**

-  We found an average `F1 Score` of `0.73` for Bidirectional-LTSM Neural Network.

<img src="https://user-images.githubusercontent.com/90170330/233785471-98042504-ebcc-4452-a876-9f8f404499ec.png" width="600">  

## Conclusion

**Barplor comparions of model performance**

<img src="https://user-images.githubusercontent.com/90170330/233785510-9b8bf20d-b9b8-4a9f-9cc4-7f09df8bd7c3.png" width="600">  

AI Predicted Tweet sentiments with `72% Accuracy`. Sentiment Analysis is a tedious task for Humans. With the aid of Machine Learning and Artificial Intelligence, we can accurately predict sentiments of a huge number amount of text, creating a stepping stone for a vast range of applications such as Social Media Monitoring and Cyber Bullying Detection.

## References
1. [https://www.kaggle.com/competitions/tweet-sentiment-extraction](https://www.kaggle.com/competitions/tweet-sentiment-extraction)
2. [https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17](https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17)
3. [https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk](https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk)
4. [https://towardsdatascience.com/intro-to-nltk-for-nlp-with-python-87da6670dde](https://towardsdatascience.com/intro-to-nltk-for-nlp-with-python-87da6670dde)
5. [https://plotly.com/python/treemaps/](https://plotly.com/python/treemaps/)
6. [https://www.analyticsvidhya.com/blog/2021/06/vader-for-sentiment-analysis/](https://www.analyticsvidhya.com/blog/2021/06/vader-for-sentiment-analysis/)
7. [https://www.enjoyalgorithms.com/blog/sentiment-analysis-using-naive-bayes/](https://www.enjoyalgorithms.com/blog/sentiment-analysis-using-naive-bayes/)
