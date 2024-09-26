#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Define filter size and dimensions
#define FILTER_SIZE 3

// Simple 3x3 filter (e.g., averaging filter)
__constant__ float filter[FILTER_SIZE * FILTER_SIZE] = {
    1.0f / 9, 1.0f / 9, 1.0f / 9,
    1.0f / 9, 1.0f / 9, 1.0f / 9,
    1.0f / 9, 1.0f / 9, 1.0f / 9
};

// CUDA kernel for applying the filter to an image
__global__ void applyFilter(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    // Calculate thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the thread is within the bounds of the image
    if (x >= width || y >= height) return;

    // Initialize the filtered pixel value
    float filteredValue = 0.0f;

    // Apply the 3x3 filter
    for (int filterY = 0; filterY < FILTER_SIZE; ++filterY) {
        for (int filterX = 0; filterX < FILTER_SIZE; ++filterX) {
            // Compute the image coordinates with boundary checking
            int imageX = min(max(x + filterX - FILTER_SIZE / 2, 0), width - 1);
            int imageY = min(max(y + filterY - FILTER_SIZE / 2, 0), height - 1);

            // Accumulate the weighted pixel value
            filteredValue += inputImage[imageY * width + imageX] * filter[filterY * FILTER_SIZE + filterX];
        }
    }

    // Write the filtered value to the output image
    outputImage[y * width + x] = (unsigned char)filteredValue;
}

// Function to load a grayscale PGM image file
bool loadPGM(const char* filename, unsigned char** image, int* width, int* height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the image file." << std::endl;
        return false;
    }

    std::string magicNumber;
    file >> magicNumber;  // P5 indicates a binary PGM file
    if (magicNumber != "P5") {
        std::cerr << "Error: Unsupported file format." << std::endl;
        return false;
    }

    // Skip comments
    file.ignore(256, '\n');
    while (file.peek() == '#') {
        file.ignore(256, '\n');
    }

    // Read image dimensions
    file >> *width >> *height;
    int maxVal;
    file >> maxVal;  // Maximum gray value (usually 255)
    file.ignore(256, '\n');  // Skip to the binary data

    // Allocate memory for the image
    *image = new unsigned char[*width * *height];

    // Read the image data
    file.read(reinterpret_cast<char*>(*image), *width * *height);
    file.close();

    return true;
}

// Function to save a grayscale PGM image file
bool savePGM(const char* filename, unsigned char* image, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the output file." << std::endl;
        return false;
    }

    // Write PGM header
    file << "P5\n" << width << " " << height << "\n255\n";

    // Write image data
    file.write(reinterpret_cast<char*>(image), width * height);
    file.close();

    return true;
}

int main() {
    const char* inputFilename = "images/bird.pgm";
    const char* outputFilename = "output/bird.pgm";

    unsigned char* h_inputImage = nullptr;
    unsigned char* h_outputImage = nullptr;
    int width, height;

    // Load the input PGM image
    if (!loadPGM(inputFilename, &h_inputImage, &width, &height)) {
        return -1;
    }

    // Allocate host memory for the output image
    h_outputImage = new unsigned char[width * height];

    // Allocate device memory for input and output images
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel to apply the filter
    applyFilter<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height);

    // Copy the output image back to the host
    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Save the processed image as a PGM file
    if (!savePGM(outputFilename, h_outputImage, width, height)) {
        return -1;
    }

    // Free device and host memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    delete[] h_inputImage;
    delete[] h_outputImage;

    std::cout << "Image processing complete. Output saved as " << outputFilename << std::endl;
    return 0;
}
