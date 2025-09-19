from rockpaperscissors import data_utils

if __name__ == "__main__":
    train_ds, val_ds = data_utils.load_data(validation_split=0.2)
    print("Train batches:", len(list(train_ds)))
    print("Val batches:", len(list(val_ds)))
