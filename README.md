\# Beam Deflection Prediction using MLP



A PyTorch-based Multi-Layer Perceptron (MLP) model to predict the deflection of a simply supported beam subjected to a central point load.



\## Model Architecture



\- Input features: `x, L, E, I, P`

\- Architecture: `5 → 64 → 64 → 1`

\- Activation: ReLU

\- Loss function: MSE

\- Optimizer: Adam

\- Epochs: 300



\## Dataset



\- 500 randomly generated beams

\- 21 points per beam

\- 10,500 total samples

\- Training/Validation/Testing split: 70/15/15



\## Test Results



| Metric | Result |

|---|---:|

| MAE | 0.148677 mm |

| RMSE | 0.434549 mm |

| Relative L2 Error | 22.21% |



\## Project Structure



```text

Beam\_MLP/

├── beam.py

├── generate\_data.py

├── preprocess.py

├── model.py

├── train.py

├── plot\_results.py

├── best\_beam\_mlp.pth

└── README.md

```



\## Installation



```bash

pip install torch numpy pandas matplotlib scikit-learn

```



\## Execution



```bash

python generate\_data.py

python preprocess.py

python train.py

python plot\_results.py

```



\## Technologies



Python | PyTorch | NumPy | Pandas | Matplotlib | Scikit-learn



\## Objective



Explore the application of neural networks as surrogate models for predicting structural beam deflection.

