# Patrick_EA_Assignment3

## Author
- Full Name: Patrick Vincent
- Assignment Title: Neural Ordinary Differential Equations (Neural ODEs) Assignment

## Description
This repository contains a solution for an assignment involving the prediction of activation times using the 2D Eikonal equation. The solution implements two neural network models:
- **Model 1**: A data-driven model using mean squared error loss.
- **Model 2**: A physics-informed model incorporating the Eikonal equation constraint.
The code uses TensorFlow 1.x for neural network training, with visualizations of true and predicted activation maps and RMSE error comparison.

## Instructions to Run the Code
1. **Prerequisites**:
   - Install Python 3.8+.
   - Install required packages: `pip install tensorflow==1.15.0 numpy scipy pyDOE matplotlib scikit-learn`.
   - Note: TensorFlow 1.x is required; ensure compatibility with your system.
2. **Run the Script**:
   - Navigate to the repository directory.
   - Run the main script: `python main.py`.
   - The script generates synthetic data, trains both models, and saves results to `model1_results.pkl` and `model2_results.pkl`.
3. **Output**:
   - Visualizations are saved as `activation_maps.png` and `rmse_error.png`.
   - Training progress and RMSE values are printed to the console.
4. **Dependencies**:
   - The code runs on CPU by default (GPU disabled via `CUDA_VISIBLE_DEVICES = "-1"`). Enable GPU if needed by removing this line (requires TensorFlow GPU support).

## Known Issues and Assumptions
- **Known Issues**:
  - Training may be slow due to the use of TensorFlow 1.x and Adam optimization. Consider using `train` (L-BFGS-B) for faster convergence if needed.
  - The script assumes sufficient memory for 100x100 grid data; reduce grid size if memory errors occur.
- **Assumptions**:
  - The Eikonal equation is simplified with a constant velocity \( V = 1 \).
  - Sparse sampling (30 points) is sufficient for the task.
  - The benchmark activation time function is appropriate for validation.

## Repository Structure
- `eikonal_model.py`: Contains the `Eikonal2DnetCV2` class implementation.
- `main.py`: Main script for data generation, training, and visualization.
- `README.md`: This file.
- (Optional) `results/`: Directory for generated plots (created during runtime).

## Submission
- Repository Link: (https://github.com/PatrickIIT/Patrick_EA_Assignment3/)
- PDF with solutions is prepared separately using LaTeX.
