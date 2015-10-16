# Flavours of Physics: the second-ranked solution

This is the solution ranked second on the [Private Leaderboard](https://www.kaggle.com/c/flavours-of-physics/leaderboard) of the Kaggle ["Flavours of Physics: Finding τ → μμμ"](https://www.kaggle.com/c/flavours-of-physics) competition. The model is based on [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) and implemented in Python with the help of the [XGBoost](https://github.com/dmlc/xgboost) library.

## Dependencies
* [XGBoost](https://github.com/dmlc/xgboost) library
* The standard Python packages **numpy**, **pandas**, and **csv** are required
* The training and test datasets (the files **training.csv** and **test.csv**) can be downloaded from [here](https://www.kaggle.com/c/flavours-of-physics/data)

## How to generate the solution
 1. Put the data files **training.csv** and **test.csv** in the **data** directory.
 2. To train the XGBoost classifiers, run **python train.py**. The trained models will be saved in the files **bst1.model** and **bst2.model**, so you can make predictions on new datasets without re-training the classifiers.
 3. To make a prediction, run **python predict.py**. Results will be written to **submission.csv**.

