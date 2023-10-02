 import numpy as np 
 import matplotlib.pyplot as plt 
 from tensorflow.keras.datasets import imdb 
 from tensorflow.keras.preprocessing.sequence import pad_sequences 
 from tensorflow.keras.models import Sequential 
 from tensorflow.keras.layers import Embedding, LSTM, GRU, SimpleRNN, Dense 
 from tensorflow.keras.utils import to_categorical 
  
 # Load the IMDb dataset 
 max_features = 10000 
 max_len = 500 
 (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) 
 x_train = pad_sequences(x_train, maxlen=max_len) 
 x_test = pad_sequences(x_test, maxlen=max_len) 
  
 # Create models 
 def create_model(model_type): 
     model = Sequential() 
     model.add(Embedding(max_features, 32, input_length=max_len)) 
     if model_type == 'LSTM': 
         model.add(LSTM(32)) 
     elif model_type == 'GRU': 
         model.add(GRU(32)) 
     elif model_type == 'RNN': 
         model.add(SimpleRNN(32)) 
     model.add(Dense(1, activation='sigmoid')) 
     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
     return model 
  
 # Train models and collect history 
 model_types = ['LSTM', 'GRU', 'RNN'] 
 histories = [] 
  
 plt.figure(figsize=(10, 6)) 
  
 for model_type in model_types: 
     model = create_model(model_type) 
     history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2, verbose=1) 
     plt.plot(history.history['accuracy'], label=f'{model_type} Accuracy') 
  
 plt.title('Comparison of Model Accuracy') 
 plt.xlabel('Epochs') 
 plt.ylabel('Accuracy') 
 plt.grid() 
 plt.legend() 
 plt.savefig("plot.png") 
 plt.show()
