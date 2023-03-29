import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import matplotlib.pyplot as plt
from skimage.color import rgb2lab


def rgb_to_lab(rgb_pixels):
    lab_pixels = rgb2lab(rgb_pixels)
    # Normalize LAB values to [0, 1] range
    lab_pixels[:, :, 0] = lab_pixels[:, :, 0] / 100.0
    lab_pixels[:, :, 1] = (lab_pixels[:, :, 1] + 128.0) / 255.0
    lab_pixels[:, :, 2] = (lab_pixels[:, :, 2] + 128.0) / 255.0
    return lab_pixels


# Load the image and convert it to a matrix of pixels
pixels = plt.imread("input_image.jpg")
# Convert RGB pixel values to LAB color space
lab_pixels = rgb_to_lab(pixels)

# Initialize PyCUDA context and device
cuda.init()
dev = cuda.Device(0)
ctx = dev.make_context()

# Define CUDA kernel for force calculation
mod = SourceModule("""
    #define EPSILON_0 8.85418782e-12

    __device__ float force(float* pixel1, float* pixel2) {
        // Calculate the CieLAB color difference between the two pixels
        float color_diff = sqrt(pow(pixel1[0]-pixel2[0], 2) + pow(pixel1[1]-pixel2[1], 2) + pow(pixel1[2]-pixel2[2], 2));
        // Calculate the distance between the two pixels
        float dist = sqrt(pow(pixel1[3]-pixel2[3], 2) + pow(pixel1[4]-pixel2[4], 2));
        // Calculate the force between the two pixels based on the rules of the simulation
        if (color_diff < 25) {
            return -EPSILON_0 / (dist * dist);
        } else {
            return EPSILON_0 / (dist * dist);
        }
    }
""")

# Define CUDA kernel for pixel update
mod = SourceModule("""
    __global__ void update_pixels(float* lab_pixels, float* forces, float* positions, int height, int width) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i >= height || j >= width) {
            return;
        }
        float* pixel_pos = positions + (i * width + j) * 2;
        float* pixel_force = forces + (i * width + j) * 2;
        // Update the pixel position based on the force acting on it
        pixel_pos[0] += pixel_force[0];
        pixel_pos[1] += pixel_force[1];
        // Enforce boundary conditions to prevent pixels from overlapping
        if (pixel_pos[0] < 0) {
            pixel_pos[0] = 0;
        } else if (pixel_pos[0] > height-1) {
            pixel_pos[0] = height-1;
        }
        if (pixel_pos[1] < 0) {
            pixel_pos[1] = 0;
        } else if (pixel_pos[1] > width-1) {
            pixel_pos[1] = width-1;
        }
        // Reset the force acting on the pixel
        pixel_force[0] = 0;
        pixel_force[1] = 0;
    }
""")

# Allocate GPU memory for pixel positions and forces
height, width, _ = lab_pixels.shape
positions = gpuarray.to_gpu(np.random.rand(height, width, 2).astype(np.float32))
forces = gpuarray.zeros((height, width, 2), dtype=np.float32)

# Define CUDA block and grid dimensions for parallel execution
block_size = (16, 16, 1)
grid_size = ((height + block_size[0] - 1) // block_size[0], (width + block_size[1] - 1) // block_size[1], 1)

# Start the simulation loop
for i in range(1000):
    # Calculate forces between neighboring pixels
    force_kernel = mod.get_function("force")
    force_kernel(cuda.In(lab_pixels), cuda.InOut(forces), cuda.In(positions), np.int32(height), np.int32(width), block=block_size, grid=grid_size)

    # Update pixel positions based on the calculated forces
    update_kernel = mod.get_function("update_pixels")
    update_kernel(cuda.InOut(positions), cuda.In(forces), np.int32(height), np.int32(width), block=block_size, grid=grid_size)

    # Retrieve the updated pixel positions from GPU memory
    updated_positions = positions.get()

    # Update the image with the new pixel positions
    updated_pixels = np.zeros_like(pixels)
    for i in range(height):
        for j in range(width):
            updated_pixels[int(updated_positions[i, j, 0]), int(updated_positions[i, j, 1]), :] = pixels[i, j, :]

    # Display the updated image
    plt.imshow(updated_pixels)
    plt.show()

# Release PyCUDA context and device
ctx.pop()
