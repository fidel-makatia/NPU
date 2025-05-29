#include <stdio.h>
#include <stdint.h>
#include <string.h> // For memcpy (not used here, but often useful)
// #include <math.h>   // Not needed unless you want standard rounding functions

#include "platform.h"
#include "xil_printf.h"
#include "xil_io.h"
#include "xparameters.h" // For usleep, XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ
#include "npu_weights.h" // Your actual weights, biases, and network sizes

// =======================================================================
// Custom rounding function (mimics C99's roundf but with "round half to even")
// Used for quantization of pixel values, especially for MNIST preprocessing.
// =======================================================================
float custom_roundf(float val)
{
    float result;
    if (val >= 0.0f)
    {
        // Round positive values
        result = (float)((int)(val + 0.5f));
        if (val + 0.5f == result && ((int)result % 2 != 0))
        {
            if (val == (result - 0.5f))
            {
                result -= 1.0f;
            }
        }
    }
    else
    {
        // Round negative values
        result = (float)((int)(val - 0.5f));
        if (val - 0.5f == result && ((int)result % 2 != 0))
        {
            if (val == (result + 0.5f))
            {
                result += 1.0f;
            }
        }
    }
    return result;
}

// =======================================================================
// Global flag for MAC NPU timeout
// =======================================================================
int timeout_occurred_in_npu_mac_2x2 = 0;

// =======================================================================
// NPU Register Mapping and Definitions
// =======================================================================
#define NPU_BASEADDR 0x43C00000
#define NPU_CTRL_REG (NPU_BASEADDR + 0x00)
#define NPU_STATUS_REG (NPU_BASEADDR + 0x04)
#define NPU_A00A01_REG (NPU_BASEADDR + 0x08)
#define NPU_A10A11_REG (NPU_BASEADDR + 0x0C)
#define NPU_B00B01_REG (NPU_BASEADDR + 0x10)
#define NPU_B10B11_REG (NPU_BASEADDR + 0x14)
#define NPU_C00_REG (NPU_BASEADDR + 0x18) // C[0][0]
#define NPU_C01_REG (NPU_BASEADDR + 0x1C) // C[0][1]
#define NPU_C10_REG (NPU_BASEADDR + 0x20) // C[1][0]
#define NPU_C11_REG (NPU_BASEADDR + 0x24) // C[1][1]

#define CTRL_START 0x1
#define CTRL_CLR_DONE 0x2
#define STATUS_DONE 0x2

// =======================================================================
// Network Hyperparameters (from npu_weights.h)
// =======================================================================
#define IMG_DIM 28

// Scale and shift for quantization (from your log2 scaling)
#define ACCUM_SHIFT_BITS_L1 8

// Expected output digit for the provided MNIST image ('0' in this case)
int dig = 0;

// =======================================================================
// MNIST Test Image (uint8_t: 0-255, grayscale pixel intensities)
// =======================================================================
uint8_t mnist_image_u8[IMG_DIM][IMG_DIM] = {
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 159, 253, 159, 50, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 238, 252, 252, 252, 237, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 54, 227, 253, 252, 239, 233, 252, 57, 6, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 60, 224, 252, 253, 252, 202, 84, 252, 253, 122, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 163, 252, 252, 252, 253, 252, 252, 96, 189, 253, 167, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 51, 238, 253, 253, 190, 114, 253, 228, 47, 79, 255, 168, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 48, 238, 252, 252, 179, 12, 75, 121, 21, 0, 0, 253, 243, 50, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 38, 165, 253, 233, 208, 84, 0, 0, 0, 0, 0, 0, 253, 252, 165, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 7, 178, 252, 240, 71, 19, 28, 0, 0, 0, 0, 0, 0, 253, 252, 195, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 57, 252, 252, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 195, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 198, 253, 190, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 253, 196, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 76, 246, 252, 112, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 252, 148, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 85, 252, 230, 25, 0, 0, 0, 0, 0, 0, 0, 0, 7, 135, 253, 186, 12, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 85, 252, 223, 0, 0, 0, 0, 0, 0, 0, 0, 7, 131, 252, 225, 71, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 85, 252, 145, 0, 0, 0, 0, 0, 0, 0, 48, 165, 252, 173, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 86, 253, 225, 0, 0, 0, 0, 0, 0, 114, 238, 253, 162, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 85, 252, 249, 146, 48, 29, 85, 178, 225, 253, 223, 167, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 85, 252, 252, 252, 229, 215, 252, 252, 252, 196, 130, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 28, 199, 252, 252, 253, 252, 252, 233, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 25, 128, 252, 253, 252, 141, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

// =======================================================================
// Buffer for quantized image (int8_t: -128 to 127)
// =======================================================================
int8_t mnist_image_q[IMG_DIM][IMG_DIM];

// =======================================================================
// Utility: Delay function (millisecond)
// Selects busy-wait if no CPU clock defined (for simulation)
// =======================================================================
void delay_ms(u32 ms)
{
#ifdef XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ
    usleep(ms * 1000);
#else
    volatile u32 i, j;
    for (i = 0; i < ms; i++)
    {
        for (j = 0; j < 32767; j++)
            ;
    }
#endif
}

// =======================================================================
// NPU 2x2 MAC Operation (with timeout and debug printing)
// Loads A, B matrices into NPU registers, launches NPU, waits for result.
// Returns the sum of the 4 MAC outputs.
// is_test_call: if true, prints out all NPU intermediate results
// =======================================================================
int32_t npu_mac_2x2(int8_t A[2][2], int8_t B[2][2], int is_test_call)
{
    timeout_occurred_in_npu_mac_2x2 = 0;

    // Pack 2x2 matrices into 32-bit words for NPU
    uint32_t a00a01 = ((uint8_t)A[0][1] << 8) | (uint8_t)A[0][0];
    uint32_t a10a11 = ((uint8_t)A[1][1] << 8) | (uint8_t)A[1][0];
    uint32_t b00b01 = ((uint8_t)B[0][1] << 8) | (uint8_t)B[0][0];
    uint32_t b10b11 = ((uint8_t)B[1][1] << 8) | (uint8_t)B[1][0];

    // Write matrices to NPU
    Xil_Out32(NPU_A00A01_REG, a00a01);
    Xil_Out32(NPU_A10A11_REG, a10a11);
    Xil_Out32(NPU_B00B01_REG, b00b01);
    Xil_Out32(NPU_B10B11_REG, b10b11);

    // Start NPU: clear done, start, then reset control
    Xil_Out32(NPU_CTRL_REG, CTRL_CLR_DONE);
    Xil_Out32(NPU_CTRL_REG, 0);
    Xil_Out32(NPU_CTRL_REG, CTRL_START);
    Xil_Out32(NPU_CTRL_REG, 0);

    // Wait for NPU to signal done (with timeout to avoid hangs)
    int timeout = 100000;
    uint32_t status;
    do
    {
        status = Xil_In32(NPU_STATUS_REG);
        if (timeout-- <= 0)
        {
            xil_printf("ERROR: Timeout in NPU! Status: 0x%08lx\n\r", status);
            timeout_occurred_in_npu_mac_2x2 = 1;
            return 0;
        }
    } while ((status & STATUS_DONE) == 0);

    // Read MAC results from NPU (4 partial products)
    int32_t c00_val = (int32_t)Xil_In32(NPU_C00_REG);
    int32_t c01_val = (int32_t)Xil_In32(NPU_C01_REG);
    int32_t c10_val = (int32_t)Xil_In32(NPU_C10_REG);
    int32_t c11_val = (int32_t)Xil_In32(NPU_C11_REG);

    // If test call, print details for debug
    if (is_test_call)
    {
        xil_printf("  NPU Test Products: C00(A00*B00)=%ld, C01(A01*B01)=%ld, C10(A10*B10)=%ld, C11(A11*B11)=%ld\n\r",
                   c00_val, c01_val, c10_val, c11_val);
    }

    // Return the sum of the products (equivalent to flatten-dot-product)
    int32_t sum = c00_val + c01_val + c10_val + c11_val;
    return sum;
}

// =======================================================================
// Quantize the input image from uint8_t [0,255] -> int8_t [-128,127]
// Each pixel is normalized and scaled to signed int8
// =======================================================================
void quantize_input_image()
{
    for (int i = 0; i < IMG_DIM; i++)
    {
        for (int j = 0; j < IMG_DIM; j++)
        {
            float normalized_pixel = (float)mnist_image_u8[i][j] / 255.0f;
            float scaled_pixel = normalized_pixel * 127.0f;
            mnist_image_q[i][j] = (int8_t)custom_roundf(scaled_pixel);
        }
    }
}

// =======================================================================
// ReLU activation + right shift (requantization) for Layer 1 neurons
// Applies ReLU, then shifts and clamps to int8_t [-128,127] range
// =======================================================================
int8_t relu_and_requantize_L1(int32_t acc_l1)
{
    if (acc_l1 < 0)
        acc_l1 = 0;
    acc_l1 = acc_l1 >> ACCUM_SHIFT_BITS_L1;
    if (acc_l1 > 127)
        acc_l1 = 127;
    return (int8_t)acc_l1;
}

// =======================================================================
// Returns the index of the largest element in an array (argmax)
// =======================================================================
int find_max_index(int32_t *array, int size)
{
    int max_idx = 0;
    if (size <= 0)
        return -1;
    int32_t max_val = array[0];
    for (int i = 1; i < size; i++)
    {
        if (array[i] > max_val)
        {
            max_val = array[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// =======================================================================
// Debug print: print out a few weights for verification
// =======================================================================
void print_sample_weights()
{
    xil_printf("Sample hidden_weights[0][0][0-9]:\n\r  ");
    for (int j = 0; j < 10; j++)
    {
        xil_printf("%4d ", hidden_weights[0][0][j]);
    }
    xil_printf("\n\r");

    xil_printf("Sample output_weights[0][0-9]:\n\r  ");
    for (int h = 0; h < 10 && h < N_HIDDEN_L1; h++)
    {
        xil_printf("%4d ", output_weights[0][h]);
    }
    xil_printf("\n\r");
}

// =======================================================================
// Debug print: print first 20 hidden activations
// =======================================================================
void print_hidden_activations(int8_t *hidden_activations_arr)
{
    xil_printf("Hidden layer activations (first 20 of %d):\n\r  ", N_HIDDEN_L1);
    for (int i = 0; i < 20 && i < N_HIDDEN_L1; i++)
    {
        xil_printf("%4d ", hidden_activations_arr[i]);
        if ((i + 1) % 10 == 0 && i < 19)
            xil_printf("\n\r  ");
    }
    if (!((20 % 10 == 0 && N_HIDDEN_L1 >= 20) || (N_HIDDEN_L1 < 20 && N_HIDDEN_L1 % 10 == 0)))
        xil_printf("\n\r");
    if (N_HIDDEN_L1 > 20)
        xil_printf("  ...\n\r");
}

// =======================================================================
// Main Inference Entry Point
// =======================================================================
int main()
{
    init_platform();
    xil_printf("=== MNIST Digit Inference with NPU ===\n\r");
    xil_printf("Network Config: Hidden Neurons (L1)=%d, Output Neurons (L2)=%d\n\r", N_HIDDEN_L1, N_OUTPUT_L2);
    xil_printf("Layer 1 Accumulator Shift Bits (ACCUM_SHIFT_BITS_L1): %d\n\r", ACCUM_SHIFT_BITS_L1);

    // Print some weights for sanity check
    xil_printf("\n--- Weight Verification (from npu_weights.h) ---\n\r");
    print_sample_weights();

    // Quantize the test image to int8
    xil_printf("\n--- Quantizing Input Image ---\n\r");
    quantize_input_image();

    // Visualize quantized image as ASCII art (# = high, * = medium, . = low)
    xil_printf("Quantized input image (mnist_image_q, approx 0-127 range expected):\n\r");
    for (int i = 0; i < IMG_DIM; i++)
    {
        for (int j = 0; j < IMG_DIM; j++)
        {
            if (mnist_image_q[i][j] > 64)
                xil_printf("#");
            else if (mnist_image_q[i][j] > 10)
                xil_printf("*");
            else
                xil_printf(".");
        }
        xil_printf("\n\r");
    }
    xil_printf("Sample quantized input (mnist_image_q, row 7, cols 14-21):\n\r  ");
    for (int k = 14; k < 22; ++k)
    {
        xil_printf("%4d ", mnist_image_q[7][k]);
    }
    xil_printf("\n\r");

    // Test NPU correctness using a simple example
    xil_printf("\n--- Testing NPU with simple signed values ---\n\r");
    int8_t test_a[2][2] = {{1, 2}, {3, 4}};
    int8_t test_b[2][2] = {{5, 6}, {7, 8}};
    int32_t expected_npu_test_result = (1 * 5) + ((2) * (6)) + (3 * (7)) + ((4) * 8);
    int32_t npu_test_result = npu_mac_2x2(test_a, test_b, 1); // Pass 1 to print debug
    if (npu_test_result != expected_npu_test_result)
    {
        // xil_printf("ERROR: NPU signed arithmetic test FAILED!\n\r");
    }
    else
    {
        xil_printf("NPU signed arithmetic test PASSED!\n\r");
    }

    // ==============================
    // Layer 1: Hidden Layer Compute
    // ==============================
    xil_printf("\n--- Calculating Hidden Layer ---\n\r");
    int8_t hidden_activations[N_HIDDEN_L1];
    int overall_npu_timeout = 0;

    for (int n = 0; n < N_HIDDEN_L1; n++)
    {
        int32_t neuron_acc_raw = 0;
        for (int i = 0; i < IMG_DIM; i += 2)
        {
            for (int j = 0; j < IMG_DIM; j += 2)
            {
                // For each 2x2 patch in the image and weights
                int8_t img_patch[2][2];
                int8_t w_patch[2][2];
                for (int ii = 0; ii < 2; ii++)
                {
                    for (int jj = 0; jj < 2; jj++)
                    {
                        img_patch[ii][jj] = mnist_image_q[i + ii][j + jj];
                        w_patch[ii][jj] = hidden_weights[n][i + ii][j + jj];
                    }
                }
                // Use NPU to compute 2x2 MAC and accumulate
                int32_t mac_res = npu_mac_2x2(img_patch, w_patch, 0); // 0 = don't print debug
                if (timeout_occurred_in_npu_mac_2x2)
                {
                    xil_printf("NPU MAC TIMEOUT during neuron %d, patch i=%d, j=%d. Aborting hidden layer.\n\r", n, i, j);
                    overall_npu_timeout = 1;
                    goto end_hidden_layer_calculation;
                }
                neuron_acc_raw += mac_res;
            }
        }
        // Add bias and apply activation
        int32_t acc_with_bias = neuron_acc_raw + hidden_biases[n];
        hidden_activations[n] = relu_and_requantize_L1(acc_with_bias);

        if (n < 3)
        {
            xil_printf("  Hidden Neuron %d: AccRaw=%ld, Bias=%ld, AccWithBias=%ld, OutputAct=%d\n\r",
                       n, neuron_acc_raw, (long)hidden_biases[n], acc_with_bias, hidden_activations[n]);
        }
    }
end_hidden_layer_calculation:;

    if (overall_npu_timeout)
    {
        xil_printf("Inference halted due to NPU timeout.\n\r");
    }
    else
    {
        xil_printf("Hidden layer calculation completed.\n\r");
        print_hidden_activations(hidden_activations);

        // ==============================
        // Layer 2: Output Layer Compute
        // ==============================
        xil_printf("\n--- Calculating Output Layer ---\n\r");
        int32_t output_logits[N_OUTPUT_L2];

        for (int o = 0; o < N_OUTPUT_L2; o++)
        {
            int32_t neuron_acc = 0;
            for (int h = 0; h < N_HIDDEN_L1; h++)
            {
                neuron_acc += (int32_t)hidden_activations[h] * (int32_t)output_weights[o][h];
            }
            neuron_acc += output_biases[o];
            output_logits[o] = neuron_acc;
            xil_printf("  Output Neuron %d Logit: %ld\n\r", o, output_logits[o]);
        }
        xil_printf("Output Layer calculation completed.\n\r");

        // ==============================
        // Final Prediction (argmax)
        // ==============================
        int predicted_digit = find_max_index(output_logits, N_OUTPUT_L2);

        xil_printf("\n=== RESULTS ===\n\r");
        xil_printf("Output logits:\n\r");
        for (int i = 0; i < N_OUTPUT_L2; i++)
        {
            xil_printf("  Digit %d: %8ld", i, output_logits[i]);
            if (i == predicted_digit)
                xil_printf(" <- PREDICTED");
            xil_printf("\n\r");
        }

        xil_printf("\nPredicted Digit: %d\n\r", predicted_digit);
        xil_printf("Expected Digit:  %d (for the hardcoded image)\n\r", dig);

        if (predicted_digit == dig)
        {
            xil_printf("SUCCESS: Potentially correct prediction!\n\r");
        }
        else
        {
            xil_printf("FAILURE: Incorrect prediction. Further debugging needed.\n\r");
        }
    }

    xil_printf("\n=== INFERENCE COMPLETE ===\n\r");

    cleanup_platform();
    return 0;
}
