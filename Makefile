# Enhanced Makefile for processing text files with various text manipulation options

# Default target
all: check_file backup delete_custom_string merge_lines remove_media_phrase remove_chars_before_third_space

# Check if file is provided and exists
check_file:
	@if [ -z $(file) ]; then \
		echo "Error: No file specified. Use 'make [target] file=FILENAME' to specify a file."; \
		exit 1; \
	elif [ ! -f $(file) ]; then \
		echo "Error: File $(file) does not exist."; \
		exit 1; \
	fi

# Backup the original file
backup: check_file
	@if [ "$(backup)" = "true" ]; then \
		echo "Creating a backup of the file..."; \
		cp $(file) $(file).bak || { echo "Error: Failed to create backup."; exit 1; }; \
	fi

# Target to delete lines containing a custom string
delete_custom_string: check_file
	@if [ -z $(string) ]; then \
		echo "Error: No custom string specified. Use 'string=CUSTOM_STRING' to specify."; \
		exit 1; \
	fi
	@echo "Deleting lines containing custom string..."
	@sed -i '/$(string)/d' $(file) || { echo "Error: Failed to delete custom string."; exit 1; }

# Target to merge lines
merge_lines: check_file
	@echo "Merging lines..."
	@awk '{if ($$0 ~ /^(Mirim:|Txai:)/) {if (NR != 1) {print line} line = $$0} else {line = line " " $$0}} END {print line}' $(file) > tmpfile && mv tmpfile $(file) || { echo "Error: Failed to merge lines."; rm -f tmpfile; exit 1; }

# Target to remove lines containing the phrase "<Arquivo de mídia oculto>"
remove_media_phrase: check_file
	@echo "Removing lines with <Arquivo de mídia oculto>..."
	@sed -i '/<Arquivo de mídia oculto>/d' $(file) || { echo "Error: Failed to remove media phrase."; exit 1; }

# Target to remove all characters before the third space
remove_chars_before_third_space: check_file
	@echo "Removing all characters before the third space..."
	@awk '{ for (i = 1; i <= NF; i++) { if (i > 3) printf "%s%s", $$i, (i < NF ? OFS : ORS) } }' $(file) > tmpfile && mv tmpfile $(file) || { echo "Error: Failed to remove characters before the third space."; rm -f tmpfile; exit 1; }

# Target to add a period at the end of lines that do not end with ',', '!', or '?'
add_period_if_needed: check_file
	@echo "Adding a period to the end of lines if needed..."
	@sed -i '/[^,!?]$$/ s/$$/./' $(file) || { echo "Error: Failed to add periods where needed."; exit 1; }

# Target to apply conversation tags to the file
tag_conversation: check_file
	@echo "Applying conversation tags..."
	@awk 'BEGIN {FS = ": "; OFS = "";} {if ($$1 == "User" || $$1 == "Txai") {print "<START><" toupper($$1) ">" $$2 "<END>"} else {print $$0}}' $(file) > $(outfile) || { echo "Error: Failed to tag conversation."; exit 1; }
	@echo "Tagging complete. Output written to $(outfile)"

# Example usage: make tag_conversation file=input.txt outfile=output.txt


# Help target to display usage
help:
	@echo "- Usage:"
	@echo "  	make [target] file=FILENAME [backup=true] [string=CUSTOM_STRING] [outfile=OUTPUT_FILENAME]"
	@echo "- Targets:"
	@echo "  	all                              - Perform all modifications on the file."
	@echo "  	delete_custom_string             - Delete lines containing a custom string in the file."
	@echo "  	merge_lines                      - Merge certain lines in the file."
	@echo "  	remove_media_phrase              - Remove lines containing a specific phrase in the file."
	@echo "  	remove_chars_before_third_space  - Remove all characters before the third space in each line."
	@echo "  	add_period_if_needed             - Add a period at the end of lines that don't end with ',', '!', or '?'."
	@echo "  	backup                           - Create a backup of the original file."
	@echo "  	tag_conversation                 - Apply model conversation tags to the file. Use 'file' for input and 'outfile' for output."
	@echo "  	help                             - Display this help message."
	@echo "- Options:"
	@echo "  	file=FILENAME                    - Specify the file to process for make targets."
	@echo "  	outfile=OUTPUT_FILENAME          - Specify the output file for the tag_conversation target."
	@echo "  	string=CUSTOM_STRING             - Specify the custom string for deletion."
	@echo "  	backup=true                      - Create a backup of the file before processing."
