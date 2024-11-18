#!/bin/bash

# Base output directory
BASE_DIR="realistic_datasets"
mkdir -p $BASE_DIR

# Log file
LOG_FILE="realistic_dataset_generation.log"
echo "Starting realistic dataset generation at $(date)" > $LOG_FILE

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

# Function to generate dataset with specific parameters
generate_dataset() {
    local name=$1
    shift
    log_message "Generating dataset: $name"
    log_message "Parameters: $@"

    python generate_dataset.py "$@" 2>&1 | tee -a $LOG_FILE

    if [ $? -eq 0 ]; then
        log_message "Successfully generated dataset: $name"
    else
        log_message "ERROR: Failed to generate dataset: $name"
    fi
    echo "----------------------------------------" >> $LOG_FILE
}

# Medium-sized base problems
log_message "Starting medium-sized problems..."

# Base medium configuration
generate_dataset "medium_base" \
    --auto \
    --num-jobs 10 --num-machines 6 \
    --min-duration 1 --max-duration 20 \
    --dependency-density 0.15 --max-deps 2 \
    --methods q_learning \
    --target-solutions 50 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 1500 \
    --max-attempts 2000

# Medium with more machines
generate_dataset "medium_more_machines" \
    --auto \
    --num-jobs 10 --num-machines 8 \
    --min-duration 1 --max-duration 20 \
    --dependency-density 0.15 --max-deps 2 \
    --methods q_learning \
    --target-solutions 50 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 1500 \
    --max-attempts 2000

# Medium with varied durations
generate_dataset "medium_varied" \
    --auto \
    --num-jobs 10 --num-machines 6 \
    --min-duration 1 --max-duration 30 \
    --dependency-density 0.15 --max-deps 2 \
    --methods q_learning \
    --target-solutions 50 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 1500 \
    --max-attempts 2000

# Larger balanced problems
log_message "Starting larger balanced problems..."

# Base large configuration
generate_dataset "large_base" \
    --auto \
    --num-jobs 15 --num-machines 8 \
    --min-duration 1 --max-duration 25 \
    --dependency-density 0.2 --max-deps 2 \
    --methods q_learning \
    --target-solutions 40 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 2000 \
    --max-attempts 2500

# Large with more exploration
generate_dataset "large_explore" \
    --auto \
    --num-jobs 15 --num-machines 8 \
    --min-duration 1 --max-duration 25 \
    --dependency-density 0.2 --max-deps 2 \
    --methods q_learning \
    --target-solutions 40 \
    --learning-rate 0.15 --discount-factor 0.95 --epsilon 0.3 --episodes 2000 \
    --max-attempts 2500

# Complex medium problems
log_message "Starting complex medium problems..."

# Medium with dependencies
generate_dataset "medium_complex" \
    --auto \
    --num-jobs 12 --num-machines 7 \
    --min-duration 1 --max-duration 25 \
    --dependency-density 0.25 --max-deps 3 \
    --methods q_learning \
    --target-solutions 45 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 1800 \
    --max-attempts 2200

# Medium-large mixed
generate_dataset "medium_large_mixed" \
    --auto \
    --num-jobs 13 --num-machines 8 \
    --min-duration 1 --max-duration 30 \
    --dependency-density 0.2 --max-deps 2 \
    --methods q_learning \
    --target-solutions 45 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 1800 \
    --max-attempts 2200

# Combined methods for medium problems
log_message "Starting combined method problems..."

# Medium combined
generate_dataset "medium_combined" \
    --auto \
    --num-jobs 10 --num-machines 6 \
    --min-duration 1 --max-duration 20 \
    --dependency-density 0.15 --max-deps 2 \
    --methods both \
    --target-solutions 60 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 1500 \
    --max-attempts 2500

# Larger combined
generate_dataset "large_combined" \
    --auto \
    --num-jobs 15 --num-machines 8 \
    --min-duration 1 --max-duration 25 \
    --dependency-density 0.2 --max-deps 2 \
    --methods both \
    --target-solutions 50 \
    --learning-rate 0.1 --discount-factor 0.95 --epsilon 0.2 --episodes 2000 \
    --max-attempts 2500

log_message "Dataset generation completed!"

# Generate summary
echo -e "\nGeneration Summary" >> $LOG_FILE
echo "===================" >> $LOG_FILE
echo "Total configurations: 9" >> $LOG_FILE
echo "Generated at: $(date)" >> $LOG_FILE
echo "Output directory: $BASE_DIR" >> $LOG_FILE
