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

3. Set up Hugging Face authentication:
   - Create a `.env` file in the project root
   - Add your Hugging Face token:
   ```
   HUGGING_FACE_HUB_TOKEN=your_token_here
   ```
   - Make sure to accept the license terms for any gated models you plan to use

## Project Structure

- `utils.py`: Core utility functions for model manipulation and steering
- `generate_completions.py`: Script for generating completions with steering vectors
- `.env`: Environment variables (not tracked in git)
- `requirements.txt`: Project dependencies

## Security Notes

- The `.env` file containing your Hugging Face token is not tracked in git
- Never commit your authentication tokens or API keys
- Make sure to keep your `.env` file secure and private