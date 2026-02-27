DATA_ROOT = "data"
IMG_SIZE = 96
BATCH_SIZE = 64
SEED = 42
CLASSES = ["rock", "paper", "scissors"]

MAKE_BEST_SUMMARY = True

EVAL_RESIZE_MODE = "pad"  # letterboxing
EVAL_RECALIB = "uniform"
RECALIB_ALPHA = 1.0
EVAL_ZERO_BIAS = True
EVAL_TTA = True
EVAL_TTA_ROT = True  # ±12°
EVAL_OUTROOT = "reports/custom_eval_myhands"
EVAL_OUTDIR_PREFIX = "my_hands_"
EVAL_ALWAYS_SUBDIR = False


TUNING = False
TUNING_MODELS = ["a", "b", "c"]  # a,b,c / model_a, model_b, model_c,
TUNING_EPOCHS = 10
TUNING_STEPS_TRAIN = None
TUNING_STEPS_VAL = None
FINAL_EPOCHS = 50
