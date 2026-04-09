from pathlib import Path

DATA_PATH = Path("data/raw/BUSI")

# Print top-level folders
print("Top-level folders:")
for p in DATA_PATH.iterdir():
    if p.is_dir():
        print("-", p.name)

print("\nSample files:\n")

# Print a few example files from each folder
for folder in DATA_PATH.iterdir():
    if folder.is_dir():
        print(f"\n{folder.name}:")
        for i, file in enumerate(folder.glob("*")):
            print(" ", file.name)
            if i >= 5:
                break