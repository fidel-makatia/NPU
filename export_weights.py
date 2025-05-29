import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
import warnings
import os
import math

# --- Configuration ---
OUTPUT_DIR = "npu_weights_c_header"
HIDDEN_NEURONS = 40
OUTPUT_NEURONS = 10
RANDOM_STATE = 42
MAX_ITER = 50 # Increase for potentially better accuracy, decrease for speed

# --- Helper Functions ---

def quantize_symmetric(data, bits=8):
    """Performs symmetric quantization on weights."""
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return data.astype(np.int8), 1.0 # Avoid division by zero

    # Calculate scale factor
    scale = (2**(bits - 1) - 1) / max_abs

    # Quantize and clip
    quantized = np.round(data * scale)
    quantized = np.clip(quantized, -128, 127)

    return quantized.astype(np.int8), scale

def quantize_bias(bias_f, scale_input, scale_weight):
    """Estimates quantized bias.
    Bias needs to match accumulator scale: float_bias / (S_input * S_weight)
    So, bias_q = float_bias * scale_input_factor * scale_weight_factor
    """
    quantized = np.round(bias_f * scale_input * scale_weight)
    return quantized.astype(np.int32)

def format_c_array_3d(data, name, dtype="int8_t"):
    """Formats a 3D numpy array into a C array string."""
    rows, cols, depth = data.shape
    c_str = f"{dtype} {name}[{rows}][{cols}][{depth}] = {{\n"
    for r in range(rows):
        c_str += "  {\n" # Start Neuron
        for c in range(cols):
            c_str += "    {" + ", ".join(map(str, data[r, c, :])) + "}"
            if c < cols - 1:
                c_str += ","
            c_str += "\n"
        c_str += "  }" # End Neuron
        if r < rows - 1:
            c_str += ","
        c_str += "\n"
    c_str += "};\n"
    return c_str

def format_c_array_2d(data, name, dtype="int8_t"):
    """Formats a 2D numpy array into a C array string."""
    rows, cols = data.shape
    c_str = f"{dtype} {name}[{rows}][{cols}] = {{\n"
    for r in range(rows):
        c_str += "  {" + ", ".join(map(str, data[r, :])) + "}"
        if r < rows - 1:
            c_str += ","
        c_str += "\n"
    c_str += "};\n"
    return c_str

def format_c_array_1d(data, name, dtype="int32_t"):
    """Formats a 1D numpy array into a C array string."""
    rows = data.shape[0]
    c_str = f"{dtype} {name}[{rows}] = {{"
    c_str += ", ".join(map(str, data))
    c_str += "};\n"
    return c_str

def save_to_header(content, filename="weights.h"):
    """Saves content to a C header file."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w") as f:
        f.write("#ifndef NPU_WEIGHTS_H_\n")
        f.write("#define NPU_WEIGHTS_H_\n\n")
        f.write('#include <stdint.h>\n\n')
        f.write(content)
        f.write("\n#endif // NPU_WEIGHTS_H_\n")
    print(f"Saved weights and biases to {filepath}")

# --- Main Script ---

# 1. Load Data
print("Fetching MNIST data...")
# Fetch MNIST dataset from OpenML, returns X (images) and y (labels)
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff', cache=True)
X = X / 255.0  # Normalize to [0, 1] - IMPORTANT for training

# 2. Train Model
print(f"Training MLPClassifier (Hidden={HIDDEN_NEURONS}, MaxIter={MAX_ITER})...")
# Create and configure the MLPClassifier (1 hidden layer)
mlp = MLPClassifier(
    hidden_layer_sizes=(HIDDEN_NEURONS,),
    max_iter=MAX_ITER,
    solver="adam",
    alpha=1e-4,
    verbose=True,
    random_state=RANDOM_STATE,
    learning_rate_init=0.001,
    early_stopping=True, # Helps prevent overfitting
    n_iter_no_change=5   # Stop if validation score doesn't improve
)

# Suppress convergence warnings during training
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
    mlp.fit(X, y) # Train on the whole dataset for final weights

print(f"Training finished. Final Score: {mlp.score(X, y):.4f}")

# 3. Extract Float Weights & Biases
# Extract weights and biases from trained model
w1_f = mlp.coefs_[0]  # Shape (784, 40)
b1_f = mlp.intercepts_[0] # Shape (40,)
w2_f = mlp.coefs_[1]  # Shape (40, 10)
b2_f = mlp.intercepts_[1] # Shape (10,)

# 4. Quantize Weights
print("Quantizing weights...")
# Quantize weights to int8 and get scale factors
w1_q, scale_w1 = quantize_symmetric(w1_f) # Shape (784, 40)
w2_q, scale_w2 = quantize_symmetric(w2_f) # Shape (40, 10)

# 5. Quantize Biases (Estimate)
print("Quantizing biases (Estimating scales)...")
# Assume input is quantized to [-128, 127], so S_i ≈ 127.0
# Assume hidden layer output is also quantized to [-128, 127], S_h ≈ 127.0
SCALE_INPUT = 127.0
SCALE_HIDDEN = 127.0
b1_q = quantize_bias(b1_f, SCALE_INPUT, scale_w1)
b2_q = quantize_bias(b2_f, SCALE_HIDDEN, scale_w2)

# 6. Reshape/Transpose for C Code
# Reshape weights for C array export:
# w1_q: (784, 40) -> (40, 28, 28) for hidden layer
w1_q_reshaped = w1_q.T.reshape((HIDDEN_NEURONS, 28, 28))
# w2_q: (40, 10) -> (10, 40) for output layer
w2_q_transposed = w2_q.T

# 7. Format C Arrays
print("Formatting C arrays...")
c_code = "// --- Generated Weights & Biases for NPU ---\n\n"
c_code += f"// Layer 1 (Input->Hidden) Weights Scale: {scale_w1:.6f}\n"
c_code += f"// Layer 2 (Hidden->Output) Weights Scale: {scale_w2:.6f}\n"
c_code += f"// Assumed Input Scale: {SCALE_INPUT:.1f}\n"
c_code += f"// Assumed Hidden Activation Scale: {SCALE_HIDDEN:.1f}\n\n"
c_code += f"#define N_HIDDEN_L1 {HIDDEN_NEURONS}\n"
c_code += f"#define N_OUTPUT_L2 {OUTPUT_NEURONS}\n\n"

# Add quantized weights and biases as C arrays
c_code += format_c_array_3d(w1_q_reshaped, "hidden_weights") + "\n"
c_code += format_c_array_1d(b1_q, "hidden_biases", dtype="int32_t") + "\n"
c_code += format_c_array_2d(w2_q_transposed, "output_weights") + "\n"
c_code += format_c_array_1d(b2_q, "output_biases", dtype="int32_t") + "\n"

# 8. Save Header File
save_to_header(c_code, "npu_weights.h")

print("\n--- Summary ---")
print(f"Layer 1 Weights Scale (scale_w1): {scale_w1:.6f}")
print(f"Layer 2 Weights Scale (scale_w2): {scale_w2:.6f}")
print("IMPORTANT: Your C code needs to handle these scales correctly.")
print("  - Input images need to be quantized (e.g., -128 to 127).")
print(f"  - Layer 1 biases were scaled assuming S_input={SCALE_INPUT:.1f}.")
print(f"  - Layer 2 biases were scaled assuming S_hidden={SCALE_HIDDEN:.1f}.")
print("  - Accumulator results need scaling/shifting before ReLU/next layer.")
print("  - The shift `>> 8` in the example C code is likely wrong and needs")
print("    to be calculated based on S_input * S_w1.")
# Estimate bit shift for accumulator scaling
shift1_est = math.log2(SCALE_INPUT * scale_w1) if SCALE_INPUT * scale_w1 > 0 else 0
shift2_est = math.log2(SCALE_HIDDEN * scale_w2) if SCALE_HIDDEN * scale_w2 > 0 else 0
print(f"  - Estimated shift for Layer 1 Acc: ~{shift1_est:.2f} bits")
print(f"  - Estimated shift for Layer 2 Acc: ~{shift2_est:.2f} bits")
print("--- Done ---")



# --- C code for quantizing input image ---
#include <math.h> // Required for roundf

// ... (other includes and defines)

// --- Quantize Input Image ---
// This function normalizes a uint8_t MNIST image to float [0,1],
// then scales to [0,127] and quantizes to int8_t, matching Python's SCALE_INPUT.
void quantize_input_image() {
    for (int i = 0; i < IMG_DIM; i++) {
        for (int j = 0; j < IMG_DIM; j++) {
            // Normalize original uint8_t [0,255] pixel to float [0,1]
            float normalized_pixel = (float)mnist_image_u8[i][j] / 255.0f;
            // Scale to [0, 127] to align with Python's SCALE_INPUT=127.0 assumption
            // for data originally in [0,1]
            float scaled_pixel = normalized_pixel * 127.0f;
            // Round and cast to int8_t. Resulting range [0, 127].
            mnist_image_q[i][j] = (int8_t)roundf(scaled_pixel);
        }
    }
}