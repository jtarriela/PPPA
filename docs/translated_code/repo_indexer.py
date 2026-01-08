import os

# CONFIGURATION
OUTPUT_FILE = "codebase_context.txt"
# Folders to ignore
IGNORE_DIRS = {'.git', '__pycache__', 'node_modules', 'dist', 'build', '.venv', 'env','materiel_f.dSYM','.claude','.github','docs/source', 'sabre','tests','verification'}
# Extensions to ignore
IGNORE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.ico', '.pyc', '.pdf', '.exe', '.bin', '.lock', '.DAT', '.sh','.99','.log','.pdf'}

# Specific files to ignore
IGNORE_FILES = {'package-lock.json', 'yarn.lock', 'flatten_repo.py', OUTPUT_FILE}

def is_text_file(filepath):
    """Simple check to see if a file is text or binary."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, IOError):
        return False

def flatten_repo():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        # Write a preamble for the LLM
        outfile.write("Below is the flattened source code for the repository.\n")
        outfile.write("Each file is delimited by '--- START OF FILE: [path] ---' and '--- END OF FILE ---'.\n\n")

        for root, dirs, files in os.walk("."):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                _, ext = os.path.splitext(file)
                if ext in IGNORE_EXTENSIONS:
                    continue

                filepath = os.path.join(root, file)
                
                # Skip binary files that slipped through extension checks
                if not is_text_file(filepath):
                    continue

                # Write the file context
                try:
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                        outfile.write(f"--- START OF FILE: {filepath} ---\n")
                        outfile.write(content)
                        outfile.write(f"\n--- END OF FILE: {filepath} ---\n\n")
                        print(f"Added: {filepath}")
                except Exception as e:
                    print(f"Skipping {filepath} due to error: {e}")

    print(f"\nSuccess! Codebase flattened into: {OUTPUT_FILE}")

if __name__ == "__main__":
    flatten_repo()