# Deep Learning MCQ VQA Submission

This project solves multiple-choice deep learning questions from PNG images using a locally run `Qwen2.5-VL-7B-Instruct` model. The inference pipeline reads a dataset folder containing `test.csv` and an `images/` directory, predicts one option per image, and writes a `submission.csv` file in the required format.

Run all commands below from the repository root, which is the directory containing:

- `inference.py`
- `requirements.txt`
- `setup.bash`
- `model`

## Repository Layout

```text
.
├── inference.py
├── requirements.txt
├── setup.bash
├── model/
├── sample_project/
│   ├── README.md
│   ├── sample_submission.csv
│   └── test.csv
└── testing_project/
    ├── images/
    ├── test.csv
    └── submission.csv
```

## System Requirements

- Linux machine with an NVIDIA GPU with atleast 48 GB of VRAM (e.g., L40S)
- Conda installed and available in the shell
- Python 3.11
- Internet access during setup only

## Environment Setup
The project is uploaded in a .zip file named project_2_25m2571_25m2573_24m2135.zip. Follow these procedures:

### Option 1: Automated setup

Step 1. unzip project_2_25m2571_25m2573_24m2135.zip

Step 2. cd project_2_25m2571_25m2573_24m2135

Use the provided script:

step 3. bash setup.bash

This script:

1. Creates a Conda environment named `gnr_project_env`
2. Installs Python 3.11
3. Installs CUDA 12.8 PyTorch
4. Installs all packages from `requirements.txt`
5. Pins `huggingface-hub` to a version compatible with the current stack
6. Downloads `Qwen/Qwen2.5-VL-7B-Instruct` into `./model` directory

Note: `setup.bash` also clones the original project repository before copying files into the current directory. If you are already inside the submitted project folder, the manual setup path below is the most direct option.

## Expected Input Structure

The inference script expects the test directory to look like this:

```text
<test_dir>/
├── test.csv
└── images/
    ├── image_1.png
    ├── image_2.png
    └── ...
```

### `test.csv` requirements

- Must contain a column named `image_name`
- Each value should be the image stem without `.png`
- For example, `image_1` maps to `images/image_1.png`

`sample_project/` contains the CSV schema example. `testing_project/` contains a runnable local example with 2 images.

## How To Run

Activate the environment and run:

Step 4. conda activate gnr_project_env

Step 5. python inference.py --test_dir /path/to/evaluation_folder

## Output

The script writes:

```text
<test_dir>/submission.csv
```

Expected output format:

```csv
id,image_name,option
image_1,image_1,3
image_2,image_2,5
```

Rules for `option`:

- `1`, `2`, `3`, `4` are valid answers
- `5` means unanswered / skipped
- Any other value is invalid

## Execution Rules

- The model must already exist in `./model` before inference starts
- Internet is not required during inference
- `inference.py` explicitly enables offline mode through:
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HUB_OFFLINE=1`
- Images must be PNG files inside `<test_dir>/images/`
- The script appends `.png` automatically to each `image_name`
- The current implementation writes output to `<test_dir>/submission.csv`
- The current script processes one image at a time.
- Although the CLI exposes `--output`, `--voting`, and `--n_votes`, the current grading path should rely only on `--test_dir`

## Important Notes

- `sample_project/` is only a format example and does not include an `images/` folder
- `testing_project/` is the correct local folder to use only for a dry run
- If an image fails during inference, the current script writes an empty prediction for that row; inspect the generated `submission.csv` before final submission
- The included code uses a local model folder and does not depend on external APIs during prediction
