import json

def process_conversation(input_file_path, output_file_path):
    # Initialize a list to hold the structured conversation
    structured_conversation = []

    # Open and read the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Process each line in the file
    for line in lines:
        # Check if the line starts with "User:" or "Polaris:"
        if line.startswith("User:") or line.startswith("Polaris:"):
            # Extract the speaker and text
            speaker, text = line.split(":", 1)
            # Append a dictionary with "from" and "text" to the list
            structured_conversation.append({"from": speaker, "text": text.strip()})

    # Only include exchanges between "User" and "Polaris"
    structured_conversation = [exchange for exchange in structured_conversation if exchange['from'] in ['User', 'Polaris']]

    # Write the structured conversation to an output file in JSON format
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(structured_conversation, outfile, ensure_ascii=False, indent=2)

# Example usage
input_file_path = 'input.txt'  # Update this to your input file path
output_file_path = 'output.json'  # Update this to your desired output file path
process_conversation(input_file_path, output_file_path)
