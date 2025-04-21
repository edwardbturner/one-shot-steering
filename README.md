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

## Dependencies

- torch: For tensor operations and neural network functionality
- nnsight: For language model manipulation
- sparsify: For sparse autoencoder functionality
- transformers: For model loading and inference
- huggingface-hub: For model access and authentication
- accelerate: For optimized model loading and inference
- python-dotenv: For environment variable management

## Usage

The project is set up to work with the Mistral-Small-24B-Instruct-2501 model, which requires authentication. Make sure you have:
1. Accepted the model's license terms on Hugging Face
2. Set up your authentication token in the `.env` file

## Security Notes

- The `.env` file containing your Hugging Face token is not tracked in git
- Never commit your authentication tokens or API keys
- Make sure to keep your `.env` file secure and private

## License

This project is licensed under the terms of your choosing. Please specify your preferred license.