#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

// Define the edge enhancement filter (Laplacian kernel)
#define FILTER_SIZE 3

__constant__ int filter[FILTER_SIZE * FILTER_SIZE] = {
     0,  -1,  0,
    -1,   5, -1,
     0,  -1,  0
};

// CUDA kernel for applying the filter to an RGB image
__global__ void applyEdgeFilter(const unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Loop through R, G, B channels
    for (int channel = 0; channel < 3; channel++) {
        int idx = (y * width + x) * 3 + channel;  // Index for the specific channel

        // Initialize the filtered pixel value
        float filteredValue = 0.0f;

        // Apply the 3x3 Laplacian filter
        for (int filterY = 0; filterY < FILTER_SIZE; ++filterY) {
            for (int filterX = 0; filterX < FILTER_SIZE; ++filterX) {
                int imageX = min(max(x + filterX - FILTER_SIZE / 2, 0), width - 1);
                int imageY = min(max(y + filterY - FILTER_SIZE / 2, 0), height - 1);

                int inputIdx = (imageY * width + imageX) * 3 + channel;
                filteredValue += inputImage[inputIdx] * filter[filterY * FILTER_SIZE + filterX];
            }
        }

        // Clamp the value to the 0-255 range
        filteredValue = min(max(filteredValue, 0.0f), 255.0f);
        outputImage[idx] = static_cast<unsigned char>(filteredValue);
    }
}

// Function to load a PPM P3 image (ASCII format)
bool loadPPM(const char* filename, unsigned char** image, int* width, int* height) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the image file." << std::endl;
        return false;
    }

    std::string magicNumber;
    file >> magicNumber;  // P3 indicates an ASCII PPM file
    if (magicNumber != "P3") {
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
    file >> maxVal;  // Maximum RGB value (usually 255)

    // Allocate memory for the image (3 channels: RGB)
    *image = new unsigned char[*width * *height * 3];

    // Read the image data (R, G, B values in ASCII format)
    for (int i = 0; i < *width * *height * 3; ++i) {
        int pixelValue;
        file >> pixelValue;
        (*image)[i] = static_cast<unsigned char>(pixelValue);
    }
    
    file.close();
    return true;
}

// Function to save a PPM P3 image (ASCII format)
bool savePPM(const char* filename, unsigned char* image, int width, int height) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the output file." << std::endl;
        return false;
    }

    // Write PPM header
    file << "P3\n" << width << " " << height << "\n255\n";

    // Write image data (R, G, B values in ASCII format)
    for (int i = 0; i < width * height * 3; ++i) {
        file << static_cast<int>(image[i]) << " ";
        if ((i + 1) % (width * 3) == 0) {
            file << "\n";
        }
    }

    file.close();
    return true;
}

int main() {
    const char* inputFilename = "images/bird2.ppm";
    const char* outputFilename = "output/bird2.ppm";

    unsigned char* h_inputImage = nullptr;
    unsigned char* h_outputImage = nullptr;
    int width, height;

    // Load the input PPM image
    if (!loadPPM(inputFilename, &h_inputImage, &width, &height)) {
        return -1;
    }

    // Allocate host memory for the output image
    h_outputImage = new unsigned char[width * height * 3];

    // Allocate device memory for input and output images
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    size_t imageSize = width * height * 3 * sizeof(unsigned char);
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    dim3 blockDim(16, 16);  // 16x16 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel to apply the edge enhancement filter
    applyEdgeFilter<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height);

    // Copy the output image back to the host
    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Save the processed image as a PPM file
    if (!savePPM(outputFilename, h_outputImage, width, height)) {
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
