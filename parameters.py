# https://github.com/gramolin/flavours-of-physics

# Random seed:
random_state = 1

# Weight for the first classifier:
w1 = 0.78

# Numbers of trees:
num_trees1 = 200 # Classifier 1
num_trees2 = 100 # Classifier 2

# Parameters of the classifiers:
params = {'objective': 'binary:logistic',
          'eta': 0.05,
          'max_depth': 4,
          'scale_pos_weight': 5.,
          'silent': 1,
          'seed': random_state}
