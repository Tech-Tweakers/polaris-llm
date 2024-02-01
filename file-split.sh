#!/bin/bash

# Check if the correct number of arguments was passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <file-to-split> <size-in-mb>"
    exit 1
fi

# Assign arguments to variables
FILE_TO_SPLIT=$1
SIZE_IN_MB=$2

# Check if the file exists
if [ ! -f "$FILE_TO_SPLIT" ]; then
    echo "Error: File '$FILE_TO_SPLIT' not found."
    exit 1
fi

# Split the file
# -d: use numeric suffixes starting at 0
# -b: size of each output file in bytes
# suffixes specify the output file names
split -d -b ${SIZE_IN_MB}m "$FILE_TO_SPLIT" "${FILE_TO_SPLIT}_part_"

echo "Splitting completed."
