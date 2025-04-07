#!/bin/bash
# WDBX CLI Demo Script
# This script demonstrates key features of the WDBX CLI

# Set up variables
DATA_DIR="./wdbx_demo_data"
EXPORT_DIR="./wdbx_export"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Helper function to show commands before executing
run_cmd() {
    echo -e "${YELLOW}$ $1${NC}"
    sleep 1
    eval "$1"
    echo ""
    sleep 0.5
}

# Helper function for section headers
section() {
    echo -e "${BLUE}========================================================${NC}"
    echo -e "${BLUE}=== $1${NC}"
    echo -e "${BLUE}========================================================${NC}"
    echo ""
}

# Start the demo
echo -e "${GREEN}Welcome to the WDBX CLI Demo!${NC}"
echo "This script will walk you through the main features of the WDBX command-line interface."
echo ""
sleep 1

# Initialize the database
section "Initialization"
run_cmd "wdbx --version"
run_cmd "wdbx init --data-dir $DATA_DIR --force"

# Create vectors
section "Creating Vectors"
# Create first vector
run_cmd "wdbx create-vector --data-dir $DATA_DIR --data \"[0.1, 0.2, 0.3, 0.4]\" --metadata '{\"description\": \"First test vector\", \"tags\": [\"test\", \"demo\"]}' --id test-vec-1 --save"

# Create more vectors
echo "Creating additional vectors..."
for i in {2..5}; do
    # Generate random vector data
    vector_data="["
    for j in {1..4}; do
        val=$(echo "scale=2; $RANDOM/32767" | bc)
        vector_data+="$val,"
    done
    vector_data="${vector_data%,}]" # Remove trailing comma
    
    # Run the command with less verbosity
    echo -e "${YELLOW}$ wdbx create-vector --data-dir $DATA_DIR --data \"$vector_data\" --metadata '{\"index\": $i}' --id test-vec-$i --save${NC}"
    wdbx create-vector --data-dir $DATA_DIR --data "$vector_data" --metadata "{\"index\": $i}" --id "test-vec-$i" --save
    echo -e "Created vector test-vec-$i\n"
done

# Search vectors
section "Searching for Similar Vectors"
run_cmd "wdbx search --data-dir $DATA_DIR --query-id test-vec-1 --top-k 3"
run_cmd "wdbx search --data-dir $DATA_DIR --query-data \"[0.15, 0.25, 0.35, 0.45]\" --top-k 3 --output-format json"

# Create blocks
section "Creating Blocks"
run_cmd "wdbx create-block --data-dir $DATA_DIR --data '{\"name\": \"First Block\", \"description\": \"Block containing test vectors\"}' --vectors '[\"test-vec-1\", \"test-vec-2\"]' --id test-block-1 --save"

run_cmd "wdbx create-block --data-dir $DATA_DIR --data '{\"name\": \"Second Block\", \"description\": \"Another test block\"}' --vectors '[\"test-vec-3\", \"test-vec-4\", \"test-vec-5\"]' --id test-block-2 --save"

# Search blocks
section "Searching Blocks"
run_cmd "wdbx search-blocks --data-dir $DATA_DIR --query-id test-vec-1 --top-k 2"
run_cmd "wdbx search-blocks --data-dir $DATA_DIR --query-text \"test block\" --top-k 2 --output-format json"

# Get objects
section "Getting Objects by ID"
run_cmd "wdbx get --data-dir $DATA_DIR --id test-vec-1 --type vector"
run_cmd "wdbx get --data-dir $DATA_DIR --id test-block-1 --type block --output-format json"

# Statistics
section "System Statistics"
run_cmd "wdbx stats --data-dir $DATA_DIR"

# Exporting data
section "Exporting Data"
run_cmd "mkdir -p $EXPORT_DIR"
run_cmd "wdbx export --data-dir $DATA_DIR --output-dir $EXPORT_DIR"
run_cmd "ls -la $EXPORT_DIR"

# Memory optimization
section "Memory Optimization"
run_cmd "wdbx optimize --data-dir $DATA_DIR"

# Clearing data and reimporting
section "Clearing and Reimporting Data"
run_cmd "wdbx clear --data-dir $DATA_DIR --confirm"
run_cmd "wdbx stats --data-dir $DATA_DIR"
run_cmd "wdbx import --data-dir $DATA_DIR --input-dir $EXPORT_DIR"
run_cmd "wdbx stats --data-dir $DATA_DIR"

# Server demonstration
section "Server Operations (demonstration only)"
echo -e "${YELLOW}$ wdbx server --data-dir $DATA_DIR --host 127.0.0.1 --port 8080 --workers 2${NC}"
echo "# Server would start here and run in the foreground"
echo "# Press Ctrl+C to stop the server"
echo ""

# Cleanup
section "Cleanup"
echo "Would you like to remove the demo data and export directories? (y/N)"
read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
    run_cmd "rm -rf $DATA_DIR $EXPORT_DIR"
    echo -e "${GREEN}Cleanup complete!${NC}"
else
    echo -e "${GREEN}Keeping data directories:${NC}"
    echo "  - $DATA_DIR"
    echo "  - $EXPORT_DIR"
fi

# Conclusion
section "Summary"
echo -e "${GREEN}This demo has shown you the main features of the WDBX CLI:${NC}"
echo "  - Initializing a database"
echo "  - Creating and managing vectors"
echo "  - Creating and managing blocks"
echo "  - Searching for similar vectors and blocks"
echo "  - Exporting and importing data"
echo "  - Optimizing memory usage"
echo "  - Starting a server (demonstrated)"
echo ""
echo -e "${GREEN}For more information, run:${NC}"
echo "  wdbx --help"
echo "  wdbx <command> --help"
echo ""
echo -e "${GREEN}Thank you for trying WDBX!${NC}" 