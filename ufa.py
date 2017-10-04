import numpy as np

# function_to_learn = lambda x: np.sin(x)
function_to_learn = lambda x: np.sin(x) + 0.1*np.random.randn(*x.shape)

np.random.seed(1000)
all_x = np.float32(
 np.random.uniform(-2*math.pi,
                    2*math.pi,
                    (1, NUM_EXAMPLES))).T
np.random.shuffle(all_x)
 
train_size = int[NUM_EXAMPLES*TRAIN_SPLIT]

trainx = all_x[:train_size]
validx = all_x[train_size:]

trainy = function_to_learn(trainx)
validy = function_to_learn(validx)
