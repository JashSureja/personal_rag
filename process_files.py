import os
import re

def extract_text(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Use regex to extract text between '00:00' and 'Transcribed by'
    text = re.search(r'00(.*?)Transcribed by', content, re.DOTALL)
    
    if text:
        return text.group(1).strip()
    else:
        return None

def process_files(directory):
    # Create a new directory to store processed files
    output_directory = os.path.join(directory, 'processed')
    os.makedirs(output_directory, exist_ok=True)

    # Process each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, filename)
            text = extract_text(file_path)
            
            if text:
                # Write the extracted text to a new file
                with open(output_path, 'w') as output_file:
                    output_file.write(text)
                print(f"Processed: {filename}")

if __name__ == "__main__":
    # Provide the directory containing your files
    directory = 'transcripts'
    process_files(directory)
