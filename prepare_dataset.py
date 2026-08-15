import os
import zipfile

def prepare_dataset(data_dir="LOLdataset", zip_path="LOL-v1.zip"):
    # 1. Extract zip if dataset folder does not exist
    if not os.path.exists(data_dir) and os.path.exists(zip_path):
        print(f"Extracting {zip_path} to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")

    low_dir = os.path.join(data_dir, "our485", "low")
    high_dir = os.path.join(data_dir, "our485", "high")

    if not os.path.exists(low_dir) or not os.path.exists(high_dir):
        print(f"Error: Could not find {low_dir} or {high_dir}")
        return

    # 15 Standard LOL validation image pairs
    eval_names = [
        '1.png', '22.png', '23.png', '55.png', '79.png', '111.png', '146.png',
        '179.png', '493.png', '547.png', '665.png', '669.png', '748.png',
        '778.png', '780.png'
    ]

    # Generate LOLv1_val.txt
    val_lines = []
    for f in eval_names:
        low_path = f"{data_dir}/our485/low/{f}"
        high_path = f"{data_dir}/our485/high/{f}"
        if os.path.exists(low_path) and os.path.exists(high_path):
            val_lines.append(f"{low_path} {high_path}")

    val_txt = os.path.join(data_dir, "LOLv1_val.txt")
    with open(val_txt, "w") as f:
        f.write("\n".join(val_lines) + "\n")
    print(f"Generated {val_txt} with {len(val_lines)} validation pairs.")

    # Generate unpaired_train.txt
    low_files = sorted(os.listdir(low_dir))
    train_lines = []
    for f in low_files:
        if os.path.exists(os.path.join(high_dir, f)):
            train_lines.append(f"{data_dir}/our485/low/{f} {data_dir}/our485/high/{f}")

    train_txt = os.path.join(data_dir, "unpaired_train.txt")
    with open(train_txt, "w") as f:
        f.write("\n".join(train_lines) + "\n")
    print(f"Generated {train_txt} with {len(train_lines)} training pairs.")

if __name__ == "__main__":
    prepare_dataset()
