# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -g -Wall -std=c++17 -I./include # Include the header files from the include folder
CUDAFLAGS = -O2 -arch=sm_60                 # Set appropriate CUDA architecture
LDFLAGS =  # Add any required linking flags

# Directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = test/output/
TEST_DIR = test
INCLUDE_DIR = include

# Source and object files for main program
SRC_CXX = $(wildcard $(SRC_DIR)/*.cpp)
SRC_CUDA = $(wildcard $(SRC_DIR)/*.cu)
OBJ_CXX = $(SRC_CXX:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ_CUDA = $(SRC_CUDA:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Source and object files for tests
TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ = $(TEST_SRC:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o) $(filter-out $(BUILD_DIR)/main.o, $(OBJ_CXX))

# Output executable names
TARGET = $(BUILD_DIR)/main
TEST_TARGET = $(BUILD_DIR)/test

# Default target to build the executable
all: $(TARGET)

test: $(TEST_TARGET)
run_tests: $(TEST_TARGET)
	@./$(TEST_TARGET)

# Linking the main object files to create the final executable
$(TARGET): $(OBJ_CXX) $(OBJ_CUDA)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile C++ files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile CUDA files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

# Test target: Compiles test files and links into a separate executable
$(TEST_TARGET): $(OBJ_CUDA) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Rule to compile test C++ files into object files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -rf $(BUILD_DIR)/*
	rm -rf $(OUTPUT_DIR)/*
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OUTPUT_DIR)

# Phony targets to prevent conflicts with files named "all" or "clean"
.PHONY: all clean
