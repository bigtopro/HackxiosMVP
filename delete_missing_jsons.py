import os
import re

# Path to the missing comments file and comments directory
missing_comments_path = "missing_comments.txt"
comments_dir = "comments"

def title_to_filename(title):
    # Replace problematic characters and spaces to match your JSON filenames
    filename = title
    filename = filename.replace("'", "")  # Remove apostrophes
    filename = filename.replace(",", "")  # Remove commas
    filename = filename.replace("&", "and")  # Replace & with and
    filename = filename.replace("(", "").replace(")", "")  # Remove parentheses
    filename = filename.replace("’", "")  # Remove curly apostrophes
    filename = filename.replace(":", "")  # Remove colons
    filename = filename.replace(".", "")  # Remove periods
    filename = filename.replace("!", "")  # Remove exclamations
    filename = filename.replace("?", "")  # Remove question marks
    filename = filename.replace("/", "_")  # Replace slashes with underscores
    filename = filename.replace("-", "_")  # Replace dashes with underscores
    filename = filename.replace(" ", "_")  # Replace spaces with underscores
    filename = re.sub(r'_+', '_', filename)  # Collapse multiple underscores
    filename = filename.strip("_")  # Remove leading/trailing underscores
    return filename + ".json"

# Step 1: Read all song titles from missing_comments.txt
with open(missing_comments_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

titles = [line.split('\t')[0] for line in lines if line.strip()]

# Step 2: Convert titles to expected JSON filenames
json_filenames = set(title_to_filename(title) for title in titles)

# Step 3: Delete matching files in the comments directory
deleted = []
for fname in os.listdir(comments_dir):
    if fname in json_filenames:
        os.remove(os.path.join(comments_dir, fname))
        deleted.append(fname)

print(f"Deleted {len(deleted)} files:")
for fname in deleted:
    print(fname) 