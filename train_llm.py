"""
Minimal LLM Training Script
A simple character-level transformer for educational purposes.
Can run on CPU or modest GPU.
"""

debug = True
monitor = True
experiment_label="4"
inputtxt = "alice.txt"


from logger import logger
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
from dataclasses import dataclass
from tokenizers_utils import CharacterTokenizer, WordTokenizer, BPETokenizer


@dataclass
class ModelConfig:
    vocab_size: int = 256  # Character-level: all ASCII characters
    context_length: int = 128  # Max sequence length
    n_layer: int = 4  # Number of transformer blocks
    n_head: int = 4  # Number of attention heads
    n_embd: int = 128  # Embedding dimension
    dropout: float = 0.1
    bias: bool = False  # Use bias in linear layers


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Q, K, V projections for all heads in a batch
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask for autoregressive generation
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length))
                            .view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dimension

        # Calculate Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble heads

        # Output projection
        y = self.resid_dropout(self.out_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying (share embeddings with output layer)
        self.token_embedding.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        logger(f"Model initialized with {n_params:,} parameters", monitor)

        logger(f"Model settings: {config}", monitor)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Token and position embeddings
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Calculate loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context to max context length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus on last token
            logits = logits[:, -1, :] / temperature
            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def get_batch(data, batch_size, context_length, device):
    """Get a random batch from the dataset"""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+1+context_length] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, data, eval_iters, batch_size, context_length, device):
    """Estimate loss over multiple batches"""
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def train(
    model,
    train_data,
    val_data,
    max_iters=5000,
    batch_size=32,
    learning_rate=3e-4,
    eval_interval=500,
    eval_iters=100,
    device='cpu',
    experiment_label="0"
):
    """Training loop"""
    logger(f"Training on {device}...", monitor)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for iter in range(max_iters):
        # Evaluate periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            train_loss = estimate_loss(model, train_data, eval_iters, batch_size,
                                      model.config.context_length, device)
            val_loss = estimate_loss(model, val_data, eval_iters, batch_size,
                                    model.config.context_length, device)
            logger(f"Step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}", monitor)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_{experiment_label}.pt")
                logger(f"Saved new best model (val loss: {val_loss:.4f})", monitor)

        # Training step
        X, Y = get_batch(train_data, batch_size, model.config.context_length, device)
        _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Progress update
        if iter % 100 == 0:
            logger(f"Step {iter}/{max_iters}, loss: {loss.item():.4f}", monitor)

    logger("Training complete!", monitor)
    return model





def main():
    # Load or create training data
    logger("Loading data...", monitor)
    try:
        with open(inputtxt, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        logger("No input text file found. Creating sample data...", monitor)
        text = """
        The quick brown fox jumps over the lazy dog.
        A journey of a thousand miles begins with a single step.
        To be or not to be, that is the question.
        All that glitters is not gold.
        """ * 100  # Repeat to have enough data
        with open(inputtxt, 'w', encoding='utf-8') as f:
            f.write(text)

    logger(f"Dataset size: {len(text):,} characters", monitor)

    # tokenizer = CharacterTokenizer(text)
    # tokenizer = WordTokenizer(text) # 3
    tokenizer = BPETokenizer.train([text], vocab_size=1000) # 4
    # tokenizer = BPETokenizer.from_pretrained("gpt2") # 5
    logger(f"Tokeniser: {tokenizer}", monitor)

    vocab_size = tokenizer.vocab_size
    logger(f"Vocabulary size: {vocab_size}", monitor)

    encode = tokenizer.encode
    decode = tokenizer.decode

    # Prepare data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Model configuration
    config = ModelConfig(vocab_size=vocab_size)
    model = SmallLLM(config)

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger(f"Using device: {device}", monitor)

    # Train
    model = train(
        model,
        train_data,
        val_data,
        max_iters=3000,
        batch_size=32,
        learning_rate=3e-4,
        device=device
    )

    # Generate sample text
    logger("\n" + "="*50, monitor)
    logger("Generating sample text:", monitor)
    logger("="*50, monitor)
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)
    logger(decode(generated[0].tolist()), monitor)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': vocab_size,
        'tokenizer': tokenizer,
    }, f"final_model_{experiment_label}.pt")
    logger(f"\nModel saved to final_model_{experiment_label}.pt", monitor)


if __name__ == '__main__':
    main()
