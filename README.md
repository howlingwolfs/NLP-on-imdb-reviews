# Sentiment Analysis on Product IMBD NLP

Welcome to my repo where I have done Sentiment Analysis and Binary Classification NLP with Tensorflow Keras on IMDB dataset

## Repository Structure
* <b>Visualization</b> <i>(Data cleaning, graphs, word cloud chart etc.)</i>
* <b>Model Training IMDB</b> <i>(Training model using keras)</i>
* <b>Predict</b> <i>(Loading and using the model with recent IMDB review)</i>
* <b>IMDB review scrapper</b> <i>(A small scrapper to check model's preformance with IMDB review of some movies)</i>

### About Data (Visualization)
[Dataset link](https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format?select=Train.csv)

There are tree files of dataset:
* Train.csv
* Test.csv
* Valid.csv

There are two columns in the CSV files. First column is <b>text</b> and the second column is <b>label</b>.
<b>Text</b> column contains all the IMDb reviews and the 
<b>label</b> column determines whether the review is positive or negative.
0 means bad review 1 means good review

<img width="441" height="473" alt="Capture" src="https://github.com/user-attachments/assets/156dd699-fab9-412d-9643-7c6d53cbfbfb" />

There are 40,000 entries in train data set and 5000 entries in test data set.
The data type of text column is object and the data type of label is int

<img width="365" height="393" alt="Capture" src="https://github.com/user-attachments/assets/af5460a2-1517-4650-bb3e-44b339866568" />

#### Checking for duplicates and null values

There are some duplicate rows but no null values in the data set

<img width="343" height="492" alt="Capture" src="https://github.com/user-attachments/assets/7926dbb6-86f6-421c-8e0e-de2d209f08b5" />

The data reflects an almost equal proportion of positive and negative reviews.

<img width="680" height="487" alt="Capture" src="https://github.com/user-attachments/assets/bf1147f8-ffed-4ecf-9625-4da9c481198c" />

#### Word cloud

This is a word cloud. This represents what word is mostly used in a data set. The largest size word is the most common word used.

#### This word cloud picture shows when the reviews are positive

<img width="935" height="470" alt="Capture" src="https://github.com/user-attachments/assets/5bf5d35c-d898-412a-b9f7-505c67c1f3d9" />

#### This word cloud picture shows when the reviews are negative

<img width="935" height="470" alt="Capture" src="https://github.com/user-attachments/assets/30a71e65-4375-4d14-be58-01c5ce10b4a7" />

### Model Training

I have trained the model with <b>tesorflow.keras</b> by using is simple dense layer model and then used LSTM.

#### Processing and tokenizing the data set
Tokenization assists in identifying and cleaning noisy or inconsistent data. For example, if you're tokenizing text data, you can easily identify words or phrases that require further preprocessing, such as stemming, lemmatization or the removal of stop words.

#### Dense Layer Model
#### A simple Keras model utilizing a Dense layer is a fundamental building block in neural networks, particularly for tasks like classification and regression
Characteristics of a Simple Dense Layer Keras Model:
* Dense Layer:
The core component is the tf.keras.layers.Dense layer. This is a fully connected layer where each neuron (or unit) in the layer is connected to every neuron in the preceding layer. This "dense" connectivity allows the layer to learn complex relationships and patterns within the input data.
* Sequential Model:
For a simple, straightforward network, the tf.keras.Sequential model is commonly used. This model allows for stacking layers in a linear fashion, where the output of one layer serves as the input to the next.
Input Layer (Implicit or Explicit):
While not always explicitly defined as a separate layer, the input shape of the first Dense layer effectively defines the input layer of the model. This input_shape parameter specifies the expected dimensions of the input data.
* Units:
The units parameter within the Dense layer defines the number of neurons in that specific layer. This directly impacts the dimensionality of the layer's output.
Activation Function:
An activation function can be applied to the output of the Dense layer. Common choices include 'relu' for hidden layers and 'softmax' for multi-class classification output layers, or 'sigmoid' for binary classification.
* Output Layer:
For tasks like classification or regression, the final Dense layer typically serves as the output layer, with the number of units corresponding to the number of classes or the dimensionality of the regression output.

#### Architecture of the dense layer model

<img width="678" height="435" alt="Capture" src="https://github.com/user-attachments/assets/e94e8bbd-15a4-4b88-a50c-b2dad691a6aa" />

<img width="334" height="167" alt="Capture" src="https://github.com/user-attachments/assets/65ec6b7b-7676-419e-941f-aab01aa0e493" />

Trained for 50 iteration. 

<img width="599" height="425" alt="Capture" src="https://github.com/user-attachments/assets/9cb347a8-095a-42b1-83b8-ab1b939bbd25" />

#### Accuracy on training set is 89%
#### Accuracy on test set is 50%

This model did not perform well as there is a lot of difference between the the accuracy test and train.
Ran the model on <b>Valid</b> data set, did not performed that well.

<img width="539" height="391" alt="Capture" src="https://github.com/user-attachments/assets/54849134-7307-49c3-a31d-697316082f3d" />

#### LSTM
LSTM (Long Short-Term Memory) Keras model refers to a neural network model built using the Keras library that incorporates one or more LSTM layers. LSTMs are a specialized type of Recurrent Neural Network (RNN) designed to overcome the vanishing gradient problem in traditional RNNs, making them particularly effective for handling sequential data and capturing long-term dependencies.
Sequential Data Handling:
LSTMs are well-suited for tasks involving sequences, such as time series forecasting, natural language processing (e.g., sentiment analysis, machine translation), and speech recognition.
Memory Cells and Gates:
Unlike simple RNNs, LSTMs include a "cell state" that acts as a long-term memory and "gates" (forget gate, input gate, output gate) that regulate the flow of information into and out of this cell state. These gates allow the model to selectively remember or forget information over long sequences.
Keras Implementation:
Keras provides a convenient LSTM layer that can be easily added to a Sequential or functional API model. This layer handles the internal complexities of LSTM cells and gates, allowing users to focus on model architecture and data preparation.
Vanishing Gradient Problem Mitigation:
The gating mechanism in LSTMs explicitly addresses the vanishing gradient problem, enabling the network to learn dependencies that span many time steps.
Building an LSTM Keras Model:
A typical LSTM Keras model often includes:
Embedding Layer (for text data):
If working with text, an Embedding layer can convert integer-encoded words into dense vector representations.
LSTM Layer(s):
The core of the model, processing the sequential input. Multiple LSTM layers can be stacked for deeper architectures.
Dense Layer(s):
Fully connected layers for outputting predictions based on the processed sequence information.
Activation Functions:
Applied to the output of layers, such as sigmoid for binary classification or softmax for multi-class classification.

#### Architecture of the dense layer model

<img width="764" height="361" alt="Capture" src="https://github.com/user-attachments/assets/82e9434b-c1bc-44c5-b4cb-3dd2361dba08" />

This model was also trained for 50 iteration. 

<img width="567" height="432" alt="download" src="https://github.com/user-attachments/assets/afe7594e-20a0-4465-88a6-dc5c14cf99a9" />

#### Accuracy on training set is 99%

#### Accuracy on test set is 83%

This model performed a lot better than the dense model. as neither overfitted or under fitted here.

<img width="540" height="451" alt="Capture" src="https://github.com/user-attachments/assets/7ebf2b24-14d2-4e60-bd1b-d19c61c09c4c" />

### Scraping IMBD reviews

Scrapping reviews from IMDB to check how our model is performing with latest reviews

<img width="578" height="202" alt="Capture" src="https://github.com/user-attachments/assets/8d475c61-5b37-4116-9ecd-9ce4d4c37700" />
<img width="1164" height="294" alt="Capture" src="https://github.com/user-attachments/assets/722a1c0a-9c80-417f-835b-d88232fefee0" />
<img width="577" height="409" alt="Capture" src="https://github.com/user-attachments/assets/b6e8e47d-2ead-400b-bfcd-eb0c16808088" />
<img width="1161" height="460" alt="Capture" src="https://github.com/user-attachments/assets/898b590e-6963-4dbc-838a-66ab2d29b081" />
<img width="1166" height="479" alt="Capture" src="https://github.com/user-attachments/assets/2d84ec26-e7d4-46f6-91dd-88719758268d" />
<img width="1165" height="305" alt="Capture" src="https://github.com/user-attachments/assets/c18c0da1-09c4-404e-9f3c-661f4d4698bf" />
<img width="1166" height="448" alt="Capture" src="https://github.com/user-attachments/assets/f3a8cd72-83c0-40fd-a46c-08c558768ca7" />
<img width="1157" height="362" alt="Capture" src="https://github.com/user-attachments/assets/fb83a53f-aa27-4c9a-9176-4f05f95addd9" />

### Imdb.keras

This is the trained saved model in keras. Its not best but still works on reviews. To use this model you will need predict.py and data files (Train.csv).

Code: 

<img width="618" height="171" alt="Capture" src="https://github.com/user-attachments/assets/df4e7e38-b184-4c2d-913d-6ad8f25be051" />












