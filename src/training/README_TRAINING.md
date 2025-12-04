# Training "Duck" (Super APL Model)

This directory contains the pipeline to train the model using the **APL/C++ Binary** dataset, filtered for a specific personality profile.

## Personality Profile: "Duck"
*   **Humor:** R2-D2 (Sassy, beeps, expressive, brave).
*   **Versatility:** C-3PO (Polyglot, protocol-obsessed, knowledgeable).

## How to Train

1.  **Install Requirements:**
    Ensure you have Python installed.

2.  **Run the Training Script:**
    ```powershell
    cd src/training
    python train_duck.py
    ```

    This script will:
    1.  Clone the source repository: `https://github.com/luckyduckcode/apl-cpp-binary-for-ai-models.git`
    2.  Apply the **R2-D2/C-3PO Filter** to the data.
    3.  Simulate the training process using the Super APL Engine settings.

## Configuration
You can adjust the personality or training parameters in `src/training/duck_personality.json`.
