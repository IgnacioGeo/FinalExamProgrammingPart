# Character-Level GPT Transformer - Final Exam Project

## Overview
This project implements a character-level GPT Transformer trained on Shakespeare's Complete Works. The model learns to generate Shakespearean-style text given a prompt.

## Files
- `models1.py`: Implementation of the scaled dot-product attention block with tests.
- `gpt_model.py`: Transformer block and GPT model implementation using the attention block.
- `train.py`: Training script including data loading, training loop, validation, checkpointing, and text generation.
- `data_prep.py`: Utility to load and preprocess text data.
- `input.txt`: Shakespeare corpus used for training.

## How to Run

1. Install dependencies:
   ```
   pip install torch numpy
   ```

2. Prepare data:
   - Ensure `input.txt` is in the project directory.

3. Train the model:
   ```
   python train.py --train
   ```

4. Generate text from a prompt:
   ```
   python train.py --generate --prompt "To be, or not to be"
   ```

## Results
- Training loss steadily decreased over 3000 iterations.
- The model generates coherent Shakespearean text.
- See the report for sample generated text and analysis.

## Notes
- Training requires a CUDA-capable GPU for reasonable speed.
- Adjust hyperparameters in `train.py` as needed.