##Handwritten Symbol Recognition

Recognizes 25 symbols.


### Packages/Libraries used:
1. scikit learn (sklearn)
2. pytorch (torch)
3. numpy (numpy)


\
<ins>The repository contains the following files:</ins>

## train&#46;py
#### Instructions to run:
        python train.py X_data.npy y_data.npy

#### Description:
This files trains our CNN model for a given input X_data(Images) and y_data(labels).

After training the model, for the input data, it saves the model to the file saved_model.pt.

#### Output:
This file prints the accuracy of the model on the training data and the validation data sets after each epoch. After the final epoch, this also prints the confusion matrix for the input dataset.

Once training is complete, it saves the model to saved_model.pt. (This file is replaced each time train. py is run. So make sure to save the model if required, before running it again.)

## saved_model&#46;pt

#### Description:
This file stores the all the parameters of the model that has been trained using the train&#46;py file above.

## test&#46;py

#### Instructions to run:
        python test.py X_data.npy y_data.npy
#### Description:
This file loads the saved_model.pt, which is the trained and saved model from train. py above, and then tests the accuracy of the model for given inputs X_data.npy(Images) and y_data.npy(Labels).

#### Output:
This file prints the accuracy of the model for the input dataset X_data and y_data, and returns a list of predicted labels for each X_data(Image). This file also prints the confusion matrix for the input dataset.

## Train_Images.npy and Train_Labels.npy
Use for training. 
Dataset is large in size and is unable to be accommodated on Github. Available on request.
