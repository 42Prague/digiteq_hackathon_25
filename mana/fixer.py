import re

IN = "Test17"

def clean_csv_line(line):
    # Split the line by semicolons
    parts = line.strip().split(';')
    
    if len(parts) != 5:
        return line.strip()  # Return original if not in expected format
    
    # Clean up 4th and 5th columns (the coordinate values)
    for i in [3, 4]:
        if i < len(parts):
            # Remove np.int64() wrapper and fix missing brackets
            if 'np.int64' in parts[i]:
                # Extract just the number using regex, handle both complete and incomplete parentheses
                match = re.search(r'np\.int64\((\d+)(?:\)?)', parts[i])
                if match:
                    number = match.group(1)
                    parts[i] = f"[{number}]"
    
    # Rejoin the cleaned parts
    return ';'.join(parts)

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            cleaned_line = clean_csv_line(line)
            outfile.write(f"{cleaned_line}\n")

if __name__ == "__main__":
    input_file = f"{IN}.csv"
    output_file = f"{IN}_cleaned.csv"
    process_file(input_file, output_file)
    print(f"Cleaned data written to {output_file}")