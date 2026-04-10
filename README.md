# Small LLM Training Project

A minimal educational implementation of a GPT-style transformer for learning how LLMs work.

## Features

- **Small & Fast**: ~50K-200K parameters depending on config
- **Character-level**: Works with any text, no tokenizer needed
- **CPU-friendly**: Trains on CPU or GPU
- **Clean code**: Well-commented, easy to understand

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For CPU-only (faster install):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Prepare Your Data

Create a file named `input.txt` with your training text. Examples:
- Your favorite book (from Project Gutenberg)
- Shakespeare's works
- Code samples
- Any text corpus you want to learn from

Or just run the script - it will create sample data if `input.txt` doesn't exist.

### 3. Train the Model

```bash
python train_llm.py
```

This will:
- Load your text data
- Initialize a small transformer model
- Train for 3000 iterations (takes 5-30 minutes depending on hardware)
- Generate sample text
- Save the trained model to `final_model.pt`

## Model Architecture

```
SmallLLM (default config)
├── Token Embedding (vocab_size × 128)
├── Position Embedding (128 × 128)
├── 4 × Transformer Blocks
│   ├── Multi-Head Attention (4 heads)
│   ├── Layer Norm
│   ├── Feed-Forward MLP
│   └── Layer Norm
└── Output Head (128 → vocab_size)

Total Parameters: ~50,000-200,000
```

## Customization

Edit the `ModelConfig` in `train_llm.py`:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 256        # Auto-detected from data
    context_length: int = 128    # Max sequence length (smaller = faster)
    n_layer: int = 4             # Number of transformer layers (2-6)
    n_head: int = 4              # Attention heads (2, 4, 8)
    n_embd: int = 128            # Embedding size (64-256)
    dropout: float = 0.1         # Regularization
```

Smaller models train faster but learn less. For limited hardware:
- Set `n_layer=2`, `n_embd=64` for faster training
- Reduce `context_length=64` to use less memory
- Lower `batch_size=16` if you run out of memory

## Understanding the Code

### Key Components

1. **MultiHeadAttention**: The core attention mechanism that lets the model focus on different parts of the input
2. **TransformerBlock**: Combines attention + feed-forward layers with residual connections
3. **SmallLLM**: The complete model that stacks transformer blocks
4. **Training Loop**: Standard gradient descent with periodic evaluation

### What Happens During Training

1. Text is converted to integers (character-level encoding)
2. Model learns to predict the next character given previous characters
3. Loss decreases as the model gets better at predictions
4. After training, it can generate new text similar to the training data

## Next Steps

Once you understand this basic implementation:

1. **Try different data**: Train on code, poetry, or dialogue
2. **Implement BPE tokenization**: Move from characters to subwords
3. **Add features**:
   - Temperature sampling
   - Beam search
   - Gradient accumulation for larger effective batch sizes
4. **Scale up**: Increase model size gradually
5. **Fine-tuning**: Load a pre-trained model and adapt it to your data

## Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Similar but more advanced implementation
- [LLM from Scratch](https://www.manning.com/books/build-a-large-language-model-from-scratch) - Detailed book

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` in the train function
- Reduce `n_embd` or `n_layer` in ModelConfig
- Reduce `context_length`

**Training loss not decreasing?**
- Check that you have enough training data (>10KB text)
- Try adjusting learning rate (1e-4 to 1e-3)
- Increase model size if it's too small

**Model generating gibberish?**
- This is normal for very small models or insufficient training
- Train longer (increase `max_iters`)
- Use more/better training data
- Increase model size

## License

MIT - Free to use for learning and experimentation
