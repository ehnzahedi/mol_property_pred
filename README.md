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
2. `data/dataset_multi.csv` is an extension of the first dataset with multi properties. P1, P2,...P9 represent the different properties to predict


## Models
  

- ### Model1
This model takes the extracted features of a molecule as input and predict the P1 property. Function fingerprint_features of the feature_extractor.py module is used to extract the features from a molecule smile. As an example:
`fingerprint_features('Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C')`

Extracted features are zero and one (similar to one-hot format). We can use both DNN or CNN to make predictions. If the extracted features are sequentioal data, CNNs would be bettre networks for predicting moleculel properties. Since we are not sure they are sequential, a custom DNN model is used for modelling.

- ## Model2
This model takes the smile string character as input and predict the `P1` property.
Sice smiles are sequential characters, a custom CNN model is implementd to get smiles and predict `P1`.  

- ## Model3
Extension of Model1 or Model2 to predict the `P1, P2,...P9` properties of the `dataset_multi.csv` dataset


#### - All the three above models are impemented as estimatores in `classifier/` . The first estimitor is designed in a way that can be trained and made prediction for both single-label and multi-label data. As a result, the first estimator can be used as `Model1` and `Model3`. 

## Flask api
To serve model1, you need to run `app.py` python file as follows.
`python app.py`
Once executed, copy the URL into a browser and it should open a web application hosted on your local machine (127.0.0.1). Try entering a molecule smile in the address bar of the browser (similar to the following) to get the prediction.
`http://127.0.0.1:5000/predict?smile=Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C`





