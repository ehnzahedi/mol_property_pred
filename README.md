# Molecule Property Prediction

This repo aims to provides deep learning models to predict molecule property.

The prediction of a drug molecule properties plays an important role in the drug design process. The molecule properties are the cause of failure for 60%
of all drugs in the clinical phases. Machine learning methods can be used to choose an optimized molecule to be subjected to more extensive studies and to avoid any clinical phase failure.



### Dataset
We have two datasets:
1. `data/dataset_single.csv` contains 4999 rows and three columns:
`smiles` : a string representation for a molecule
`mol_id` : Unique id for the molecule
`P1` : binary property to predict
2. `data/dataset_multi.csv` is an extension of the first dataset with multi properties. P1, P2,...P9 represent the different properties that can be predicted.

### Data Exploration and Pre-Processing
Check this 
[Jupyter Notebook](https://github.com/ehnzahedi/mol_property_pred/blob/dev/data_exploration_preprocessing.ipynb).


## Models
  

- ### Model1
This model takes the extracted features of a molecule as input and predict the P1 property. Function fingerprint_features of the feature_extractor.py module is used to extract the features from a molecule smile. As an example:
`fingerprint_features('Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C')`

Extracted features are zero and one (similar to one-hot format). We can use both DNN or CNN to make predictions. If the extracted features are sequentioal data, CNNs would be bettre networks for predicting moleculel properties. Since we are not sure they are sequential, a custom DNN model is used for modelling.

  - For model1, `FeatureExtractedClassifier` estimator can be set similar to:
  
  `clf = FeatureExtractedClassifier(epochs=100,
                                 batch_size=16,
                                 metrics=METRICS,
                                 optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                                 class_weight=class_weight
                                 )`
 The lebels are imbalanced. So, we use `class_weight` to train Model1 and use AUC, F1-score, recall, ... to measure the performance of the model.     
  

- ## Model2
This model takes the smile string character as input and predict the `P1` property.
Sice smiles are sequential characters, a custom CNN model is implementd to get smiles and predict `P1`.  

  - For model2, `SmilePredictor` estimator can be set similar to:
  
  `clf = SmilePredictor(epochs=2,
                      batch_size=16,
                      loss='binary_crossentropy',
                      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,
                                                        momentum=0.0,
                                                        nesterov=False),
                      metrics=METRICS,
                      class_weight=class_weight,
                      validation_split=0.2
                      )`

- ## Model3
Extension of Model1 to predict the `P1, P2,...P9` properties of the `dataset_multi.csv` dataset.

  - For Model3, the first estimator is used:
  `clf = eatureExtractedClassifier(epochs=5,
                                 batch_size=16,
                                 metrics=METRICS,
                                 loss='binary_crossentropy',
                                 output_activation='sigmoid',
                                 optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                                 class_weight=class_weight,
                                 is_multi_label=True
                                 )`


#### - All the three above models are impemented as estimatores in `classifier/` . The first estimitor is designed in a way that can be trained and made prediction for both single-label and multi-label data. As a result, the first estimator can be used as `Model1` and `Model3`. 

## Example
To better use the deep learning models (estimators) and evaluate them, please see [this](https://github.com/ehnzahedi/mol_property_pred/blob/dev/example.ipynb) notebook.

## Main
The `main.py` allows users to train and evaluate a model as well as predicting the property `P1` for a given smile. Try entering `python main.py` in command line with these arguments:

`python main.py model1 train`: train model1 on the train set of `data/dataset_single.csv` dataset.

`python main.py model1 evaluate`: evaluate model1 on the test set of `data/dataset_single.csv` dataset.

`python main.py model1 predict Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C`: a smile can be pass to the trained model1 to predict `P1`.

`model1` can be replaced in the above commands in order to train (and evaluate) model3 and also predict `P1, P2,...P9` properties for a given smile.


## Evaluation
During the training a fraction of training data is used for validation. This fraction can be set by `validation_split` in the estimators. Two function in [utils.py](https://github.com/ehnzahedi/mol_property_pred/blob/dev/utils.py) are used to plot and calculate the desired metric for validation set and test set. 

Since `data/dataset_multi.csv` is multi label, [`hamming_loss`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html), in addition to other metrics, is used to validate the performance of Model3. 



## Flask api
To serve model1, you need to run `app.py` python file as follows.
`python app.py`
Once executed, copy the URL into a browser and it should open a web application hosted on your local machine (127.0.0.1). Try entering a molecule smile in the address bar of the browser (similar to the following) to get the prediction.

`http://127.0.0.1:5000/predict?smile=Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C`


## Packaging
The application can be installable using `setup.py`.





