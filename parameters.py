# https://github.com/gramolin/flavours-of-physics

# Random seed:
random_state = 1

# Weight for the first booster:
w1 = 0.78

# Numbers of trees:
num_trees1 = 200 # Booster 1
num_trees2 = 100 # Booster 2

# Parameters of the first booster:
params1 = {'objective': 'binary:logistic',
           'eta': 0.05,
           'max_depth': 4,
           'scale_pos_weight': 5.,
           'silent': 1,
           'seed': random_state}

# Parameters of the second booster:
params2 = {'objective': 'binary:logistic',
           'eta': 0.05,
           'max_depth': 4,
           'scale_pos_weight': 5.,
           'silent': 1,
           "seed": random_state}
