
import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split



# 1. LOAD DATASET


df = pd.read_csv("beam_dataset.csv")

print("Original dataset shape:", df.shape)



# 2. SPLIT BY BEAM ID


SEED = 42

# Get unique beam IDs
beam_ids = df["beam_id"].unique()

# First split: 70% training, 30% remaining
train_ids, remaining_ids = train_test_split(
    beam_ids,
    train_size=0.70,
    random_state=SEED,
    shuffle=True
)

# Second split: divide remaining 30% equally
val_ids, test_ids = train_test_split(
    remaining_ids,
    test_size=0.50,
    random_state=SEED,
    shuffle=True
)

# Select all rows belonging to each beam group
train_df = df[df["beam_id"].isin(train_ids)].copy()
val_df = df[df["beam_id"].isin(val_ids)].copy()
test_df = df[df["beam_id"].isin(test_ids)].copy()



# 3. VERIFY THE SPLIT


print("\nDATASET SPLIT")
print("----------------------------")

print("Training beams:", len(train_ids))
print("Validation beams:", len(val_ids))
print("Testing beams:", len(test_ids))

print("\nTraining rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Testing rows:", len(test_df))



# 4. SAVE ORIGINAL SPLITS


train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)



# 5. DEFINE INPUT AND OUTPUT COLUMNS


input_columns = ["x", "L", "E", "I", "P"]
output_column = ["y"]



# 6. CALCULATE TRAINING STATISTICS


X_train = train_df[input_columns].to_numpy()
y_train = train_df[output_column].to_numpy()

# Calculate mean and standard deviation
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)



# 7. NORMALIZATION FUNCTION


def normalize(data, mean, std):
    return (data - mean) / std


# Normalize training data
train_df.loc[:, input_columns] = normalize(
    train_df[input_columns].to_numpy(),
    X_mean,
    X_std
)

train_df.loc[:, output_column] = normalize(
    train_df[output_column].to_numpy(),
    y_mean,
    y_std
)

# Normalize validation data using TRAINING statistics
val_df.loc[:, input_columns] = normalize(
    val_df[input_columns].to_numpy(),
    X_mean,
    X_std
)

val_df.loc[:, output_column] = normalize(
    val_df[output_column].to_numpy(),
    y_mean,
    y_std
)

# Normalize testing data using TRAINING statistics
test_df.loc[:, input_columns] = normalize(
    test_df[input_columns].to_numpy(),
    X_mean,
    X_std
)

test_df.loc[:, output_column] = normalize(
    test_df[output_column].to_numpy(),
    y_mean,
    y_std
)



# 8. SAVE NORMALIZATION STATISTICS


scaler_stats = {
    "input_columns": input_columns,
    "output_columns": output_column,
    "X_mean": X_mean.tolist(),
    "X_std": X_std.tolist(),
    "y_mean": y_mean.tolist(),
    "y_std": y_std.tolist()
}

with open("scaler_stats.json", "w") as file:
    json.dump(scaler_stats, file, indent=4)



# 9. SAVE NORMALIZED DATASETS


train_df.to_csv("train_normalized.csv", index=False)
val_df.to_csv("validation_normalized.csv", index=False)
test_df.to_csv("test_normalized.csv", index=False)



# 10. FINAL VERIFICATION


print("\nNORMALIZATION COMPLETED")
print("----------------------------")

print("\nTraining input means:")
print(train_df[input_columns].mean())

print("\nTraining input standard deviations:")
print(train_df[input_columns].std(ddof=0))

print("\nTraining output mean:")
print(train_df["y"].mean())

print("\nTraining output standard deviation:")
print(train_df["y"].std(ddof=0))

print("\nFINAL DATASET SHAPES")
print("----------------------------")

print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Testing:", test_df.shape)

print("\nSaved all datasets and scaler statistics!")


