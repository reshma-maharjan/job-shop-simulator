import os

def convert_data(input_file, output_file):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create full paths for input and output files
    input_path = os.path.join(script_dir, input_file)
    output_path = os.path.join(script_dir, output_file)
    
    with open(input_path, 'r') as f:
        # Skip the first line containing dimensions
        dimensions = f.readline().strip()
        rows, cols = map(int, dimensions.split())
        
        # Read the data lines
        data = []
        for line in f:
            # Convert each line into pairs of numbers
            numbers = list(map(int, line.split()))
            # Split into pairs: [(machine, time), (machine, time),...]
            pairs = [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]
            data.append(pairs)
        
        # Separate times and machines into two matrices
        times = [[pair[1] for pair in row] for row in data]
        machines = [[pair[0] + 1 for pair in row] for row in data]  # Add 1 to convert from 0-based to 1-based
        
        # Write to output file
        with open(output_path, 'w') as out_f:
            # Write dimensions
            out_f.write(f"{rows}  {cols}\n")
            
            # Write times matrix
            for row in times:
                out_f.write('\t'.join(map(str, row)) + '\n')
            
            # Write machines matrix
            for row in machines:
                out_f.write('\t'.join(map(str, row)) + '\n')

# File names
input_file = 'la40_old.txt'
output_file = 'la40.txt'

convert_data(input_file, output_file)