import numpy as np
from sklearn.datasets import fetch_openml

def generate_mnist_c_array(digit_to_find=7):
    """
    Fetches MNIST, finds an image of the specified digit,
    and returns it as a C-style uint8_t 2D array string.
    """
    print(f"Fetching MNIST data (this might take a moment)...")
    try:
        # Load MNIST data (pixel values 0-255)
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='liac-arff', cache=True)
        print("Data loaded successfully.")

        # Convert labels to integers (from string)
        y = y.astype(np.uint8)

        # Find indices for the desired digit
        digit_indices = np.where(y == digit_to_find)[0]

        if len(digit_indices) == 0:
            # No image found for the specified digit
            return f"Error: No digit '{digit_to_find}' found in the dataset."

        # Select the first image found for the digit
        image_index = digit_indices[0]
        image_flat = X[image_index]

        # Reshape flat image to 28x28 and ensure uint8 type
        image_2d = image_flat.reshape(28, 28).astype(np.uint8)

        # Format as C array
        print(f"Formatting C array for digit '{digit_to_find}'...")
        rows, cols = image_2d.shape
        c_str = f"uint8_t mnist_image_u8_{digit_to_find}[{rows}][{cols}] = {{\n"
        for row in image_2d:
            # Format each number with padding for alignment
            c_str += "    {" + ", ".join([f"{val:3d}" for val in row]) + "},\n"
        c_str = c_str.rstrip(",\n") + "\n};"

        return c_str

    except Exception as e:
        # Handle any errors during data fetching or processing
        return f"An error occurred: {e}"

# --- Main Execution ---
# Set the digit you want to extract (e.g., 0)
target_digit = 0

# Generate and print the C array code for the specified digit
c_array_output = generate_mnist_c_array(target_digit)

print("\n" + "="*50)
print(f"    C Array for MNIST Digit '{target_digit}'")
print("="*50)
print(c_array_output)
print("="*50)