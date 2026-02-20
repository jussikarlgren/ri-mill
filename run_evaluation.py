"""
Example script demonstrating how to evaluate your trained LLM.

This script loads a trained model and evaluates its text generation quality.
"""

import torch
from train_llm import SmallLLM, ModelConfig
from evaluate_llm import LLMEvaluator, evaluate_model_outputs
from logger import logger
debug = True
monitor = True


def load_model(model_path: str = 'best_model.pt', device: str = 'cpu'):
    """Load a trained model from checkpoint."""
    logger(f"""Loading model {model_path}""", monitor)
    checkpoint = torch.load(model_path, map_location=device)
    logger(f"""Loaded model {model_path}""", monitor)

    # Handle both checkpoint formats:
    # - best_model_*.pt saves state_dict directly
    # - final_model_*.pt saves a dict with 'model_state_dict' key
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = ModelConfig(
            vocab_size = checkpoint.get('vocab_size', 256),
            context_length = checkpoint.get('context_length', 128),
            n_layer = checkpoint.get('n_layer', 4),
            n_head = checkpoint.get('n_head', 4),
            n_embd = checkpoint.get('n_embd', 128),
            dropout = checkpoint.get('dropout', 0.1)
        )
    else:
        # Checkpoint is the state_dict directly, use default config
        state_dict = checkpoint
        config = ModelConfig()

    model = SmallLLM(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, config


def main():
    """Main evaluation script."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger(f"Using device: {device}", debug)

    model_path = "best_model.pt"

    # Load the trained model
    logger("Loading trained model...", debug)
    try:
        model, config = load_model(model_path, device)
        logger("Model loaded successfully!", debug)
    except FileNotFoundError:
        print(f"""Error: {model_path} not found. Please train the model first using train_llm.py""")
        return

    # Load training data for reference comparison (optional)
    reference_text = None
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            reference_text = f.read()
        logger(f"Loaded reference corpus ({len(reference_text)} characters)", debug)
    except FileNotFoundError:
        logger("No reference corpus found (input.txt). Evaluation will proceed without reference comparison.", debug)

    # Set up character encoding/decoding
    if reference_text:
        chars = sorted(list(set(reference_text)))
    else:
        # Use ASCII characters as fallback
        chars = [chr(i) for i in range(256)]

    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def decode(indices):
        """Convert token indices to text."""
        return ''.join([idx_to_char.get(idx, '?') for idx in indices])

    # Evaluation configurations
    logger("\n" + "="*60, debug)
    logger("EVALUATION CONFIGURATIONS", debug)
    logger("="*60, debug)
    logger(f"Number of samples: 5", debug)
    logger(f"Max tokens per sample: 500", debug)
    logger(f"Reference comparison: {'Yes' if reference_text else 'No'}", debug)
    logger("="*60, debug)

    # Run evaluation
    results = evaluate_model_outputs(
        model=model,
        decode_fn=decode,
        num_samples=5,
        max_tokens=500,
        reference_corpus=reference_text
    )

    # Save detailed results
    resultfile = "evaluation_results.txt"
    logger(f"""\nSaving detailed results to {resultfile}...""", debug)
    with open(resultfile, 'w', encoding='utf-8') as f:
        f.write("LLM EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")

        for result in results:
            f.write(f"\nSAMPLE {result['sample_id']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Generated Text:\n{result['text']}\n\n")

            metrics = result['metrics']
            f.write(f"Overall Score: {metrics['overall_score']['overall']:.2f}/100 ({metrics['overall_score']['grade']})\n")
            f.write(f"Diversity Score: {metrics['diversity']['diversity_score']:.4f}\n")
            f.write(f"Repetition Ratio: {metrics['diversity']['repetition_ratio']:.4f}\n")
            f.write(f"Type-Token Ratio: {metrics['diversity']['type_token_ratio']:.4f}\n")
            f.write("\n")

    logger("Results saved successfully!", debug)

    # Quick evaluation of single generation
    logger("\n" + "="*60, debug)
    logger("QUICK SINGLE SAMPLE EVALUATION", debug)
    logger("="*60, debug)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=300)[0].tolist()
    sample_text = decode(generated)

    logger(f"\nGenerated sample:\n{sample_text}\n", debug)

    evaluator = LLMEvaluator(reference_corpus=reference_text)
    evaluator.evaluate(sample_text, verbose=True)


if __name__ == "__main__":
    main()
