CXX = g++ 
CXXFLAGS = -std=c++17 -Wall -Wextra -O0 -fopenmp -w

# Directories
SRC_DIR = src
INCLUDE_DIR = include
LIB_DIR = lib
TEST_DIR = test
BIN_DIR = bin

# Files
MAIN_SRC = $(SRC_DIR)/main.cpp
LIB_SRCS = $(wildcard $(LIB_DIR)/*.cpp)

HEADERS = $(wildcard $(INCLUDE_DIR)/*.hpp)
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_EXEC = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))

# Targets
MAIN_EXEC = $(BIN_DIR)/app

.PHONY: all clean tests sandbox float finetune_dense_scale

all: $(MAIN_EXEC) tests

$(MAIN_EXEC): $(MAIN_SRC) $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $(MAIN_SRC) $(LIB_SRCS)

$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LIB_SRCS)

float: $(SRC_DIR)/get_online_perf_float.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(BIN_DIR)/get_online_perf_float $(SRC_DIR)/get_online_perf_float.cpp $(LIB_SRCS)
	@echo "Running float model simulation..."
	@./$(BIN_DIR)/get_online_perf_float || echo "Float model simulation execution failed with exit code $$?"
	
finetune_dense_scale: $(SRC_DIR)/finetune_dense_scale.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(BIN_DIR)/finetune_dense_scale $(SRC_DIR)/finetune_dense_scale.cpp $(LIB_SRCS)
	@echo "Running finetune_dense_scale..."
	@./$(BIN_DIR)/finetune_dense_scale || echo "Finetune gradient scale execution failed with exit code $$?"

sandbox: $(SRC_DIR)/sandbox.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(BIN_DIR)/sandbox $(SRC_DIR)/sandbox.cpp $(LIB_SRCS)
	@echo "Running sandbox..."
	@./$(BIN_DIR)/sandbox || echo "Sandbox execution failed with exit code $$?"

clean:
	rm -rf $(BIN_DIR)

# Run all tests with better output and parallel execution
tests:clean $(TEST_EXEC)
	@echo "Running unit tests..."
	@echo "Total tests: $(words $(TEST_EXEC))"
	@echo "--------------------------------------"
	@printf "%s\n" $(TEST_EXEC) | xargs -I{} -P $(shell nproc) bash -c 'echo Running {}; ./{} || echo "Test {} failed"'
	@echo "--------------------------------------"
	@echo "All tests completed!"

# Parallel compilation
$(TEST_EXEC): $(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LIB_SRCS)

$(BIN_DIR)/%: $(SRC_DIR)/%.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	echo "Compiling $<"
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LIB_SRCS)

.PHONY: parallel_compile

parallel_compile: $(TEST_SRCS)
	@echo "Compiling tests in parallel..."
	@printf "%s\n" $(TEST_SRCS) | xargs -I{} -P $(shell nproc) $(MAKE) $(BIN_DIR)/$(notdir {})
