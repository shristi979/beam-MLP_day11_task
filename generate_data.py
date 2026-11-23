
import numpy as np
import pandas as pd

from beam import ss_center_point


# 1. CONFIGURATION

SEED = 72
NUM_BEAMS = 500
POINTS_PER_BEAM = 21

rng = np.random.default_rng(SEED)



# 2. GENERATE RANDOM BEAM PARAMETERS

rows = []

for beam_id in range(NUM_BEAMS):

    # Random beam properties
    L = rng.uniform(1.0, 5.0)          # metres
    E = rng.uniform(70e9, 210e9)       # Pa
    I = rng.uniform(1e-6, 1e-4)        # m^4
    P = rng.uniform(100, 5000)        # N

    # Store beam properties in dictionary
    beam = {
        "L": L,
        "E": E,
        "I": I,
        "P": P
    }


    
    # 3. GENERATE 21 POSITIONS
    

    positions = np.linspace(0, L, POINTS_PER_BEAM)


    
    # 4. CALCULATE DEFLECTION
    

    for x in positions:

        y = ss_center_point(x, beam)

        rows.append({
            "beam_id": beam_id,
            "x": x,
            "L": L,
            "E": E,
            "I": I,
            "P": P,
            "y": y
        })



# 5. CREATE DATAFRAME


df = pd.DataFrame(rows)



# 6. SAVE DATASET


df.to_csv("beam_dataset.csv", index=False)



# 7. VERIFY DATASET


print("\nDATASET GENERATED SUCCESSFULLY!")
print("--------------------------------")

print("Total beams:", NUM_BEAMS)
print("Points per beam:", POINTS_PER_BEAM)
print("Total rows:", len(df))

print("\nDataset shape:", df.shape)

print("\nFirst 10 rows:")
print(df.head(10))

print("\nDataset statistics:")
print(df.describe())

print("\nNumber of unique beams:", df["beam_id"].nunique())

print("\nSaved as: beam_dataset.csv")


