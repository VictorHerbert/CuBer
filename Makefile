# Compiler and flags
NVCC = nvcc
CXX = $(NVCC)
CXXFLAGS_LK = -G -g -O0 -std=c++17 -I./include
CXXFLAGS = $(CXXFLAGS_LK) -dc -MD -MP  \
  -gencode=arch=compute_52,code=sm_52 \
  -gencode=arch=compute_75,code=sm_75 \
  -gencode=arch=compute_86,code=sm_86
LDFLAGS =  # Optional linker flags

# Directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = test/output
TEST_DIR = test
INCLUDE_DIR = include

# Source files
SRC = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
OBJ := $(OBJ:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Test files
TEST_SRC = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJ = $(TEST_SRC:$(TEST_DIR)/%.cpp=$(BUILD_DIR)/%.o) $(filter-out $(BUILD_DIR)/main.o, $(OBJ))

# Targets
TARGET = $(BUILD_DIR)/main
TEST_TARGET = $(BUILD_DIR)/test

# Default target
all: $(TARGET)

test: $(TEST_TARGET)

run_tests: $(TEST_TARGET)
	@./$(TEST_TARGET)

# Link main
$(TARGET): $(OBJ)
	$(NVCC) $(CXXFLAGS_LK) -o $@ $^

# Link test
$(TEST_TARGET): $(TEST_OBJ)
	$(NVCC) $(CXXFLAGS_LK) -o $@ $^

# Compile rules with dependency generation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)/*
	rm -rf $(OUTPUT_DIR)/*
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OUTPUT_DIR)

# Include auto-generated dependency files
-include $(OBJ:.o=.d)

.PHONY: all clean test run_tests
