# Patrick_EA_Assignment3

## Author
- Full Name: Patrick Vincent
- Assignment Title: Neural Ordinary Differential Equations (Neural ODEs) Assignment

## Description
This repository contains a solution for a Neural ODE assignment that compares a standard 1-hidden-layer neural network with a Neural ODE model for binary classification on the MNIST dataset (digits 0 and 1, downsampled to 8x8 pixels). The solution includes:
- A `StandardNet` with one hidden layer (16 neurons, ReLU activation).
- An `ODENet` using a Neural ODE block with `torchdiffeq` and the `dopri5` solver.
- Training scripts, accuracy reporting, and visualization of loss and confusion matrices.
- Performance comparison and discussion as per Part C requirements.

## Instructions to Run the Code
1. **Prerequisites**:
   - Install Python 3.8+.
   - Install required packages: `pip install torch torchvision scikit-learn matplotlib seaborn pydoe scipy torchdiffeq`.
2. **Run the Script**:
   - Navigate to the repository directory.
   - Run the main script with default settings: `python main.py`.
   - To specify a model, use: `python main.py --model standard` or `python main.py --model odenet`.
   - Adjust epochs, batch size, or learning rate with `--epochs`, `--batch_size`, and `--lr` flags.
3. **Output**:
   - Training loss, accuracy, and visualizations (`standard_results.png`, `odenet_results.png`) will be generated.
4. **Dependencies**:
   - Ensure a GPU is available (optional) by setting `--gpu 0`; otherwise, it defaults to CPU.

## Known Issues and Assumptions
- **Known Issues**:
  - The script assumes the MNIST dataset can be downloaded. If download fails, ensure internet connectivity or pre-download the dataset.
  - Visualization may require sufficient memory; adjust `batch_size` if memory errors occur.
- **Assumptions**:
  - The binary MNIST subset (0 and 1) is sufficient for the task.
  - The `dopri5` solver is adequate for the Neural ODE; other solvers (e.g., `rk4`) could be explored.
  - High accuracy (near 99.9%) is expected due to the simplicity of the binary classification task.

## Repository Structure
- `main.py`: Main Python script with the implementation.
- `README.md`: This file.
- (Optional) `results/`: Directory for generated plots (created during runtime).

## Submission
- Repository Link: (https://github.com/PatrickIIT/Patrick_EA_Assignment3/)
- PDF with solutions is prepared separately using LaTeX.
