
CXX = g++ 
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fopenmp

# Directories
SRC_DIR = src
INCLUDE_DIR = include
LIB_DIR = lib
TEST_DIR = test
BIN_DIR = bin

# Files
MAIN_SRC = $(SRC_DIR)/main.cpp
LIB_SRCS = $(LIB_DIR)/*.cpp

HEADERS = $(INCLUDE_DIR)/*.hpp
TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
TEST_EXEC = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TEST_SRCS))

# Targets
MAIN_EXEC = $(BIN_DIR)/app

.PHONY: all clean tests sandbox

all: $(MAIN_EXEC) tests

$(MAIN_EXEC): $(MAIN_SRC) $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $(MAIN_SRC) $(LIB_SRCS)

$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $@ $< $(LIB_SRCS)

sandbox: $(SRC_DIR)/sandbox.cpp $(LIB_SRCS) $(HEADERS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -o $(BIN_DIR)/sandbox $(SRC_DIR)/sandbox.cpp $(LIB_SRCS)
	@echo "Running sandbox..."
	@./$(BIN_DIR)/sandbox || echo "Sandbox execution failed with exit code $$?"

clean:
	rm -rf $(BIN_DIR)

# Run all tests
tests: $(TEST_EXEC)
	@for test in $(TEST_EXEC); do \
	  echo "Running $$test"; \
	  ./$$test || exit 1; \
	done
