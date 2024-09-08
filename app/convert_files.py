import docx2txt
import glob
import os

# Define the directory containing .docx files
directory = './all-transcripts/'

# Get the list of .docx files in the directory
docx_files = glob.glob(directory + '*.docx')

# Create the 'transcripts' directory if it doesn't exist
transcripts_directory = 'transcripts'
os.makedirs(transcripts_directory, exist_ok=True)

# Iterate over each .docx file
for docx_file in docx_files:
    # Extract the file name without extension
    file_name = os.path.splitext(os.path.basename(docx_file))[0]

    # Generate the output .txt file path in the 'transcripts' directory
    txt_file = os.path.join(transcripts_directory, file_name + '.txt')

    # Convert .docx to plain text and save to .txt file
    with open(docx_file, 'rb') as infile:
        with open(txt_file, 'w', encoding='utf-8') as outfile:
            doc_text = docx2txt.process(infile)
            outfile.write(doc_text)

print("All .docx files converted to .txt format and stored in the 'transcripts' directory.")
