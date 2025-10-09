DATA_ROOT = "data"
IMG_SIZE = 96
BATCH_SIZE = 64
SEED = 42
CLASSES = ["rock", "paper", "scissors"]

TUNING = True
TUNING_MODELS = ["c"] # choose one or more models
TUNING_FAST = True
TUNING_EPOCHS = 10
TUNING_STEPS_TRAIN = None
TUNING_STEPS_VAL = None
FINAL_EPOCHS = 50
NO_TUNING = True # switch to True to avoid tuning