import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

from model import BeamMLP

# Load training history
history = pd.read_csv("training_history.csv")

# Extract data
epochs = history["epoch"]
train_losses = history["train_loss"]
val_losses = history["val_loss"]

print("Training history loaded successfully!")
print("Total epochs:", len(history))
print("Best validation loss:", val_losses.min())
print("Best epoch:", val_losses.idxmin() + 1)

# Create learning curve
plt.figure(figsize=(10, 6))

plt.plot(
    epochs,
    train_losses,
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (Normalized)")
plt.title("Beam MLP Training and Validation Loss")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig("learning_curve.png", dpi=300)
plt.show()

print("Learning curve saved successfully!")


# ==========================================
# 1. LOAD BEST MODEL
# ==========================================

model = BeamMLP()

checkpoint = torch.load(
    "best_beam_mlp.pth",
    map_location="cpu"
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nBest model loaded successfully!")
print("Best epoch:", checkpoint["epoch"])
print("Best validation loss:", checkpoint["best_val_loss"])


# ==========================================
# 2. LOAD TEST DATA
# ==========================================

test_normalized = pd.read_csv("test_normalized.csv")
test_original = pd.read_csv("test.csv")

input_columns = ["x", "L", "E", "I", "P"]

X_test = test_normalized[input_columns].values

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# ==========================================
# 3. LOAD SCALER STATISTICS
# ==========================================

with open("scaler_stats.json", "r") as file:
    scaler_stats = json.load(file)

y_mean = scaler_stats["y_mean"][0]
y_std = scaler_stats["y_std"][0]

print("\nTest samples:", len(X_test))


# ==========================================
# 4. MODEL PREDICTIONS
# ==========================================

with torch.no_grad():

    predictions_normalized = model(X_test_tensor).numpy()

# Convert normalized predictions back to meters
y_pred = predictions_normalized * y_std + y_mean

# Actual deflections from original test dataset
y_true = test_original["y"].values.reshape(-1, 1)

# Convert meters to millimeters
y_pred_mm = y_pred * 1000
y_true_mm = y_true * 1000

# Calculate prediction errors
errors_mm = y_pred_mm - y_true_mm

print("\nPredictions generated successfully!")

# ==========================================
# 5. EVALUATION METRICS
# ==========================================

mae = np.mean(np.abs(errors_mm))

rmse = np.sqrt(np.mean(errors_mm ** 2))

relative_l2 = (
    np.linalg.norm(y_pred - y_true)
    / np.linalg.norm(y_true)
)

print("\n===================================")
print("FINAL TEST RESULTS")
print("===================================")

print(f"Test MAE: {mae:.6f} mm")
print(f"Test RMSE: {rmse:.6f} mm")
print(f"Relative L2 Error: {relative_l2:.6f}")
print(f"Relative L2 Error: {relative_l2 * 100:.4f}%")


# ==========================================
# 6. PREDICTED VS ACTUAL
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_true_mm,
    y_pred_mm,
    alpha=0.5,
    label="Predictions"
)

# Perfect prediction reference line
min_val = min(y_true_mm.min(), y_pred_mm.min())
max_val = max(y_true_mm.max(), y_pred_mm.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Deflection (mm)")
plt.ylabel("Predicted Deflection (mm)")
plt.title("Predicted vs Actual Beam Deflection")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("predicted_vs_actual.png", dpi=300)
plt.show()

# ==========================================
# 7. ERROR HISTOGRAM
# ==========================================

plt.figure(figsize=(8, 6))

plt.hist(
    errors_mm.flatten(),
    bins=40,
    edgecolor="black",
    alpha=0.75
)

plt.xlabel("Prediction Error (mm)")
plt.ylabel("Frequency")
plt.title("Distribution of Beam Deflection Prediction Errors")

plt.axvline(
    0,
    linestyle="--",
    label="Zero Error"
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("error_histogram.png", dpi=300)
plt.show()


# ==========================================
# 8. THREE TEST BEAM COMPARISON
# ==========================================

plt.figure(figsize=(10, 6))

# Select three different test beams
beam_groups = test_original.groupby("beam_id")

selected_beams = list(beam_groups.groups.keys())[:3]

for beam_id in selected_beams:

    # Get integer row positions belonging to this beam
    indices = np.flatnonzero(
        test_original["beam_id"].to_numpy() == beam_id
    )

    # Extract actual positions
    x_values = test_original.iloc[indices]["x"].to_numpy()

    # Extract actual and predicted deflections
    actual = y_true_mm[indices].flatten()
    predicted = y_pred_mm[indices].flatten()

    # Sort according to beam position
    order = np.argsort(x_values)

    x_values = x_values[order]
    actual = actual[order]
    predicted = predicted[order]

    # Plot actual deflection
    plt.plot(
        x_values,
        actual,
        marker="o",
        label=f"Beam {beam_id} - Actual"
    )

    # Plot predicted deflection
    plt.plot(
        x_values,
        predicted,
        linestyle="--",
        marker="x",
        label=f"Beam {beam_id} - Predicted"
    )

plt.xlabel("Position Along Beam (m)")
plt.ylabel("Deflection (mm)")
plt.title("Actual vs Predicted Deflection for Three Test Beams")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("three_beam_comparison.png", dpi=300)
plt.show()

print("\nAll test visualizations saved successfully!")
