**Data Visualization**

[Qualtrics Data Analysis competition](https://github.com/roshan151/Qualtrics-competition/blob/main/Qualtrics_2nd_Prize.pptm) was organized by Qualtrics organization at UT Dallas business school. The goal of the competition was to visulaize company wide data including sales, support, marketing and customer data to dervice insights that can boost sales or improve customer retention. I cleaned and organized the data using Python scripts and visulaized them using Power BI. 20% of qualtrics customers were generating 90% of totla profit, on analyzing their characterstical differences it was concluded that these premium customers tend to have a dedicated and small team for Qualtrics projects while customers generating low profit had this reponsibility divided over various teams. Another major insight was that their was no prioritization in resolving the support requests of these premium customers leading to a relatively greater resolve times. My team of four students from UT Dallas grabbed second place in a pool of 120 participats. 

**Regression, Decision Tree and SVM**

[Appllied Machine Learning Project](https://github.com/roshan151/roshan151/blob/main/AML_Project%20Code.ipynb) utlised data from 512 different species of mushrooms to classify them as poisonous or edible based on their characterstics like cap length, cap shape, color , stem length etc. Data was cleaned, visualized and 12 different models were tested to select the best performing models. It was implemented by a group of four students and I executed Logistic Regression, Polynomial Logistic Regression at degree 3, Decision Trees and Support Vector Machine. It could be internpreted from the scatter plot of data points that SVM is best suited for the dataset and it is confirmed by the F1 score. By adjusting the threshold of sigmoid function of logistic regression I was able to reduce the false negatives predicted by the model which is crucial for the final result as false positives represented the poisonous mushrooms as edible.   

**Computer Vision**

[Multiclass Classification](https://github.com/roshan151/roshan151/blob/main/Multiclass_classification_for_Computer_Vision.ipynb): In this poject I have used a convolutional neural network model to classifies images with hand signals into 25 different classes. Model architecture consists of 2 convolutional and max pooling layers, 1 dropout layer to avoid overfitting, 1 dense layer with 514 neurons and a final dense layer with 25 neurons and softmax. 

[Transfer Learning](https://github.com/roshan151/roshan151/blob/main/Transfer_Learning_for_Computer_Vision.ipynb): Here I have created  a computer vision model which uses imported weights from a model trained on extensive dataset- InceptionV3. All the layers of InceptionV3 model are set to untrainable except for the last. Then a final DNN architecture is added with a dense layer of 1024 neuron, a dropout layer and a final dense layer with 1 neuron and sigmoid activation to classify colored images of humans and horses.

**Time Series Analysis using Tensorflow**

[Moving Average Baseline model](https://github.com/roshan151/roshan151/blob/main/Time_series_creating_data.ipynb):In this notebook I have created an artificial time series by adding gradual slope, seasonal variations over a fixed period and random noise to the graph. Then I have used an average of previous few predictions to predict the next value and this has created a moving average time series prediction baseline model. 

[LSTM Model to outperform baseline](https://github.com/roshan151/roshan151/blob/main/LSTM_to_predict_Time_Series.ipynb): This model outperforms the moving average baseline in predicting the time series. It uses 2 bidirectional LSTMs layers to learn the features of the timeseries sending context to and fro and then predicts the next value of a time series.

[Prediction of sun spot temperatures](https://github.com/roshan151/roshan151/blob/main/Sunspots_Time_Series_Prediction.ipynb): Sunspot Temperatures is a huge dataset that records temperature of sun's surface as far back as 1980's. My deep learning architecture aims at predicting this time series using a convolutional layer, two LSTM layers and three Dense layers. It also uses a Stochastic Gradient Descent optimizer which selects the best learning rate by running the model on a range of learning rates. 

**Natural Language Processing**

[Sarcasm Detection](https://github.com/roshan151/roshan151/blob/main/NLP%20Sarcasm%20detection.ipynb): This project highlights the importance of GLove embeddings in Natural Lanaguage Processing. It trains an architecture of one Bidirectional GRU - Gated Recurrent Unit, one dense layer with 32 neurons and a final dense layer with 1 neuron and sigmoid activation. This model is trained twice once without the weights of GLove embeddings and again with it. The difference is substanial and the model without GLove embeddings tends to drastically overfit. 

[Text Generator](https://github.com/roshan151/roshan151/blob/main/Text_Generator_NLP.ipynb): In this notebook I train a Bidirectional LSTM to predict the next word in a sequence. For the training set I have used Shakespeare's sonnets and I have fed each sentence multiple times, each time removing one word from the sequence and using that word as the predicted label. Finally the model tries to create a poem by accepting a seed text as input. 



