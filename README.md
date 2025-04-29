# One-Shot Steering

This project implements one-shot steering techniques for language models using sparse autoencoders (SAEs) and transcoders.

## Setup
```bash
pip install uv
uv venv
uv sync
```

## Overview

- `generate_code_steering_vectors.py` or `generate_medical_steering_vectors.py`: Use this to generate your steering vector
- `eval.py`: Then use this to generate completions with
   the model + your steering vector

## Credit

- Ideas all from [Jacob's post](https://www.lesswrong.com/posts/kcKnKHTHycHeRhcHF/one-shot-steering-vectors-cause-emergent-misalignment-too)
- Code forked from [Caden's implementation](https://github.com/cadentj/steering)

## Requirements
- 1xH100