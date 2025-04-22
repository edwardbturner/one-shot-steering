# One-Shot Steering

This project implements one-shot steering techniques for language models using sparse autoencoders (SAEs) and transcoders.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Overview

- `generate_steering_vectors.py`: Use this to generate your steering vector
- `eval.py`: Then use this to generate completions with
   the model + your steering vector

## Credit

- Ideas all from [Jacob's post](https://www.lesswrong.com/posts/kcKnKHTHycHeRhcHF/one-shot-steering-vectors-cause-emergent-misalignment-too)
- Code forked from [Caden's implementation](https://github.com/cadentj/steering)

## Requirements
- 1xH100