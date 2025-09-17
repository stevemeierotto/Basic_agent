import os
import sys

def read_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading {filepath}: {e}"

def collect_headers(project_dir):
    include_dir = os.path.join(project_dir, 'include')
    if not os.path.isdir(include_dir):
        return []
    # Only include .h files, ignore subdirectories
    headers = sorted([
        f for f in os.listdir(include_dir)
        if f.endswith('.h') and os.path.isfile(os.path.join(include_dir, f))
    ])
    return [os.path.join(include_dir, f) for f in headers]

def main(project_dir='.'):
    header_files = collect_headers(project_dir)
    output_path = os.path.join(project_dir, 'headers.txt')
    with open(output_path, 'w') as outfile:
        outfile.write("=== HEADER FILES ===\n")
        for filepath in header_files:
            outfile.write(f"\n{filepath}:\n```cpp\n")
            outfile.write(read_file(filepath))
            outfile.write("\n```\n")
    print(f"Header files written to {output_path}")

if __name__ == "__main__":
    project_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(project_dir)

