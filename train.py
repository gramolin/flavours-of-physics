# https://github.com/gramolin/flavours-of-physics

import pandas, xgboost, features, parameters

# Read the training dataset:
train = pandas.read_csv('data/training.csv', index_col='id')
train = train[train['min_ANNmuon'] > 0.4]

# Add extra features:
train = features.add_features(train)

# Train the first (geometric) XGBoost classifier:
bst1 = xgboost.train(parameters.params,
                     xgboost.DMatrix(train[features.list1],
                     train['signal']), parameters.num_trees1)
bst1.save_model('bst1.model')

# Train the second (kinematic) XGBoost classifier:
bst2 = xgboost.train(parameters.params,
                     xgboost.DMatrix(train[features.list2],
                     train['signal']), parameters.num_trees2)
bst2.save_model('bst2.model')
