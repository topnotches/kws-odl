#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>

// Function prototype (ensure it matches the implementation you're testing)
int32_t requantize_shift(int32_t large_number, const float rescale_value, const uint8_t activation_bits);

// Test case structure
struct TestCase {
    int32_t large_number;
    float rescale_value;
    uint8_t activation_bits;
    int32_t expected_result;
};

// Function to run test cases
void run_tests() {
    std::vector<TestCase> test_cases = {
        {0, 1.0, 8, 1},     
        {10, .5, 8, 5},     
        {255, 2.2, 9, 510},     
        {255, 3.2, 9, 511},     
        {255, 1.2, 9, 255},     
        {255, 3.1, 12, 1020},    
        {0, 1.0, 8, 1},
        {4, 1.0, 8, 5),
        {255, 1.0, 8, 255},
        {128, 0.5, 8, 65},
        {64, 2.0, 8, 128},
        {512, 0.25, 8, 129},
        {1024, 0.125, 8, 129},
        {-11000, 0.1, 8, 0},
        {10, 10.0, 8, 80},
        {256, 0.01, 8, 3},
        {999, 0.75, 8, 255},
        {0, 1.0, 8, 1},
        {255, 1.0, 8, 255},
        {1000, 0.1, 8, 125},
        {512, 0.25, 8, 129},
        {1024, 0.125, 8, 129},
        {10, 10.0, 8, 80},
        {256, 0.01, 8, 3},
        {999, 0.75, 8, 255},
        {4000, 0.2, 8, 255},
        {32768, 0.001, 8, 33}
    };

    bool all_passed = true;

    for (const auto& test : test_cases) {
        int32_t result = requantize_shift(test.large_number, test.rescale_value, test.activation_bits);
        if (result != test.expected_result) {
            std::cerr << "Test failed: "
                      << "requantize_shift(" << test.large_number << ", " 
                      << test.rescale_value << ", " << (int)test.activation_bits
                      << ") = " << result << " (expected " << test.expected_result << ")\n";
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "All tests passed successfully!\n";
    } else {
        std::cerr << "Some tests failed.\n";
    }
}

// Main function to execute tests
int main() {
    run_tests();
    return 0;
}
