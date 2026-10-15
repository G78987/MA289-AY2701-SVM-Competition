# MA289-AY2701-SVM-Competition

README: 

The problem and prediction goal:The goal of this project was to predict whether or not the audio file said six or seven. The audio file was represented as a spectrogram, and so the modeling task was to differentiate spectrograms so as to recognize sixes and sevens. 

The model(s) used: In order to do this, I used a SVM with a Radial Basis Function and Histogram of Oriented Gradients. This method allowed me to draw out differences in the training data set and use a infinite dimensions to separate the data.

Your validation approach: To validate my approach, I used cross validation. This allowed me to test my findings without reducing the training data amount.

Your final Kaggle score: 0.90646

Instructions for reproducing your results: In order to reproduce my work, download the test and submission files. The training data can be found at the following link: https://www.kaggle.com/competitions/ma-289-spoken-digit-data-competition/data, it is too large to include in a github repository. Run the attached code file in Google Colab. The code will automatically download the output prediction to your computer as an excel file.
