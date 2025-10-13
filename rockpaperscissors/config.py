DATA_ROOT = "data"
IMG_SIZE = 96
BATCH_SIZE = 64
SEED = 42
CLASSES = ["rock", "paper", "scissors"]

EVAL_RESIZE_MODE = "pad"     # letterboxing coerente
EVAL_RECALIB = "uniform"     # 'off' | 'uniform' | 'empirical'
RECALIB_ALPHA = 1.0          # 0.7–1.2 per modulare l’effetto
EVAL_ZERO_BIAS = True        # rimuove il prior appreso nell’ultimo Dense
EVAL_TTA_ROT = True          # True se vuoi anche le rotazioni ±12°
EVAL_OUTROOT = "reports/custom_eval_myhands"  # base directory per i report
EVAL_OUTDIR_PREFIX = "my_hands_"              # prefisso della sottocartella
EVAL_ALWAYS_SUBDIR = False                    # True => crea sempre /<prefix><tag> dentro --outdir


TUNING = True
TUNING_MODELS = ["c"] # choose one or more models
TUNING_FAST = True
TUNING_EPOCHS = 10
TUNING_STEPS_TRAIN = None
TUNING_STEPS_VAL = None
FINAL_EPOCHS = 50
NO_TUNING = True # switch to True to avoid tuning