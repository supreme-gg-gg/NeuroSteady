import csv

# Define file names
input_filename = "data.txt"  # Input text file
output_filename = "dataset.csv"  # Output CSV file

# Open the input text file and read its contents
with open(input_filename, mode="r") as file:
    lines = file.readlines()  # Read all lines

# Open the output CSV file in write mode
with open(output_filename, mode="w", newline="") as file:
    writer = csv.writer(file)

    # Write column headers (modify if needed)
    writer.writerow(["AccelX", "AccelY", "AccelZ", "ButtonState"])

    # Write dataset rows
    for line in lines:
        cleaned_line = line.replace(",", " ")  # Remove commas
        row = cleaned_line.strip().split()  # Split values by spaces
        writer.writerow(row)  # Write to CSV

print(f"CSV file '{output_filename}' saved successfully!")
