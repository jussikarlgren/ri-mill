"""
LLM Output Evaluation Scheme

This module provides automated metrics for evaluating text generation quality
without requiring ground truth data. Suitable for assessing language model outputs
based on linguistic and statistical properties.
"""

import torch
import numpy as np
from collections import Counter
import re
from typing import List, Dict, Any
import math


class LLMEvaluator:
    """Evaluates generated text from language models using automated metrics."""

    def __init__(self, reference_corpus: str = None):
        """
        Args:
            reference_corpus: Optional reference text (e.g., training data) for comparison
        """
        self.reference_corpus = reference_corpus
        if reference_corpus:
            self.reference_stats = self._compute_corpus_stats(reference_corpus)

    def evaluate(self, generated_text: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Comprehensive evaluation of generated text.

        Args:
            generated_text: The text to evaluate
            verbose: Whether to print results

        Returns:
            Dictionary containing all evaluation metrics
        """
        metrics = {}

        # 1. Basic Statistics
        metrics['basic'] = self._basic_statistics(generated_text)

        # 2. Diversity Metrics
        metrics['diversity'] = self._diversity_metrics(generated_text)

        # 3. Coherence Metrics
        metrics['coherence'] = self._coherence_metrics(generated_text)

        # 4. Quality Indicators
        metrics['quality'] = self._quality_indicators(generated_text)

        # 5. Reference Comparison (if available)
        if self.reference_corpus:
            metrics['reference_comparison'] = self._reference_comparison(generated_text)

        # 6. Overall Score
        metrics['overall_score'] = self._compute_overall_score(metrics)

        if verbose:
            self._print_results(metrics)

        return metrics

    def _basic_statistics(self, text: str) -> Dict[str, float]:
        """Compute basic text statistics."""
        words = text.split()
        chars = list(text)

        return {
            'total_chars': len(chars),
            'total_words': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_chars': len(set(chars)),
            'unique_words': len(set(words)),
        }

    def _diversity_metrics(self, text: str) -> Dict[str, float]:
        """
        Measure vocabulary diversity and repetition.
        Higher diversity usually indicates better generation.
        """
        words = text.split()
        chars = list(text)

        # Type-Token Ratio (TTR) - ratio of unique words to total words
        ttr = len(set(words)) / len(words) if words else 0

        # Character diversity
        char_diversity = len(set(chars)) / len(chars) if chars else 0

        # Bigram diversity
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0

        # Word bigrams
        word_bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
        word_bigram_diversity = len(set(word_bigrams)) / len(word_bigrams) if word_bigrams else 0

        # Repetition ratio (how much text is repeated)
        repetition_ratio = self._calculate_repetition(text)

        return {
            'type_token_ratio': ttr,
            'char_diversity': char_diversity,
            'bigram_diversity': bigram_diversity,
            'word_bigram_diversity': word_bigram_diversity,
            'repetition_ratio': repetition_ratio,
            'diversity_score': (ttr + bigram_diversity + word_bigram_diversity) / 3
        }

    def _coherence_metrics(self, text: str) -> Dict[str, float]:
        """
        Measure text coherence through structural patterns.
        """
        # Sentence detection (simple heuristic)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Average sentence length
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0

        # Sentence length variance (too uniform might indicate problems)
        sentence_lengths = [len(s.split()) for s in sentences]
        sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0

        # Punctuation usage
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        punctuation_ratio = punctuation_count / len(text) if text else 0

        return {
            'num_sentences': len(sentences),
            'avg_sentence_length': avg_sentence_length,
            'sentence_length_variance': sentence_length_variance,
            'punctuation_ratio': punctuation_ratio,
        }

    def _quality_indicators(self, text: str) -> Dict[str, Any]:
        """
        Indicators of generation quality issues.
        """
        # Repeated sequences (n-grams that appear multiple times)
        repeated_3grams = self._find_repeated_ngrams(text, n=3)
        repeated_4grams = self._find_repeated_ngrams(text, n=4)

        # Consecutive character repetition (like "aaaa")
        max_char_repeat = max([len(match.group()) for match in re.finditer(r'(.)\1+', text)], default=0)

        # All caps ratio (shouting detection)
        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0

        # Digit ratio
        digit_ratio = sum(1 for c in text if c.isdigit()) / len(text) if text else 0

        # Special character ratio
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if text else 0

        return {
            'repeated_3grams': len(repeated_3grams),
            'repeated_4grams': len(repeated_4grams),
            'max_char_repetition': max_char_repeat,
            'caps_ratio': caps_ratio,
            'digit_ratio': digit_ratio,
            'special_char_ratio': special_ratio,
        }

    def _reference_comparison(self, text: str) -> Dict[str, float]:
        """
        Compare generated text to reference corpus.
        """
        gen_stats = self._compute_corpus_stats(text)
        ref_stats = self.reference_stats

        # Compare distributions
        kl_divergence = self._kl_divergence(gen_stats['char_freq'], ref_stats['char_freq'])

        # Vocabulary overlap
        gen_words = set(text.split())
        ref_words = set(self.reference_corpus.split())
        vocab_overlap = len(gen_words & ref_words) / len(gen_words) if gen_words else 0

        # Style similarity (based on word length distribution)
        style_similarity = 1 - abs(gen_stats['avg_word_len'] - ref_stats['avg_word_len']) / max(gen_stats['avg_word_len'], ref_stats['avg_word_len'])

        return {
            'kl_divergence': kl_divergence,
            'vocab_overlap': vocab_overlap,
            'style_similarity': style_similarity,
        }

    def _compute_overall_score(self, metrics: Dict) -> Dict[str, float]:
        """
        Compute an overall quality score (0-100).
        Higher is better.
        """
        scores = []

        # Diversity component (0-100)
        diversity = metrics['diversity']['diversity_score'] * 100
        repetition_penalty = (1 - metrics['diversity']['repetition_ratio']) * 100
        scores.append((diversity + repetition_penalty) / 2)

        # Quality component (penalty for issues)
        quality_score = 100
        quality_score -= min(metrics['quality']['repeated_3grams'] * 5, 50)  # Penalize repetitions
        quality_score -= min(metrics['quality']['max_char_repetition'] * 10, 30)  # Penalize char repeats
        scores.append(max(quality_score, 0))

        # Coherence component
        coherence_score = 100
        if metrics['coherence']['avg_sentence_length'] < 3:
            coherence_score -= 30  # Too short sentences
        elif metrics['coherence']['avg_sentence_length'] > 50:
            coherence_score -= 20  # Too long sentences
        scores.append(max(coherence_score, 0))

        overall = np.mean(scores)

        return {
            'overall': overall,
            'diversity_component': scores[0],
            'quality_component': scores[1],
            'coherence_component': scores[2],
            'grade': self._score_to_grade(overall)
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90: return 'A (Excellent)'
        elif score >= 80: return 'B (Good)'
        elif score >= 70: return 'C (Fair)'
        elif score >= 60: return 'D (Poor)'
        else: return 'F (Very Poor)'

    def _calculate_repetition(self, text: str, window: int = 20) -> float:
        """Calculate what fraction of text is repeated sequences."""
        if len(text) < window:
            return 0.0

        sequences = [text[i:i+window] for i in range(len(text) - window + 1)]
        sequence_counts = Counter(sequences)

        # Count sequences that appear more than once
        repeated = sum(count - 1 for count in sequence_counts.values() if count > 1)
        total = len(sequences)

        return repeated / total if total > 0 else 0.0

    def _find_repeated_ngrams(self, text: str, n: int) -> List[str]:
        """Find n-grams that appear multiple times."""
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        ngram_counts = Counter(ngrams)
        return [ng for ng, count in ngram_counts.items() if count > 1]

    def _compute_corpus_stats(self, text: str) -> Dict:
        """Compute statistical properties of a corpus."""
        words = text.split()
        chars = list(text)

        # Character frequency distribution
        char_counts = Counter(chars)
        total_chars = sum(char_counts.values())
        char_freq = {char: count/total_chars for char, count in char_counts.items()}

        return {
            'char_freq': char_freq,
            'avg_word_len': np.mean([len(w) for w in words]) if words else 0,
        }

    def _kl_divergence(self, p: Dict, q: Dict) -> float:
        """Compute KL divergence between two probability distributions."""
        all_chars = set(p.keys()) | set(q.keys())

        kl = 0.0
        for char in all_chars:
            p_val = p.get(char, 1e-10)
            q_val = q.get(char, 1e-10)
            kl += p_val * math.log(p_val / q_val)

        return kl

    def _print_results(self, metrics: Dict):
        """Pretty print evaluation results."""
        print("\n" + "="*60)
        print("LLM OUTPUT EVALUATION RESULTS")
        print("="*60)

        print("\n📊 BASIC STATISTICS")
        print("-" * 60)
        for key, value in metrics['basic'].items():
            print(f"  {key:25s}: {value:>10.2f}")

        print("\n🎨 DIVERSITY METRICS")
        print("-" * 60)
        for key, value in metrics['diversity'].items():
            print(f"  {key:25s}: {value:>10.4f}")

        print("\n📝 COHERENCE METRICS")
        print("-" * 60)
        for key, value in metrics['coherence'].items():
            print(f"  {key:25s}: {value:>10.2f}")

        print("\n⚠️  QUALITY INDICATORS")
        print("-" * 60)
        for key, value in metrics['quality'].items():
            if isinstance(value, float):
                print(f"  {key:25s}: {value:>10.4f}")
            else:
                print(f"  {key:25s}: {value:>10}")

        if 'reference_comparison' in metrics:
            print("\n🔍 REFERENCE COMPARISON")
            print("-" * 60)
            for key, value in metrics['reference_comparison'].items():
                print(f"  {key:25s}: {value:>10.4f}")

        print("\n⭐ OVERALL SCORE")
        print("-" * 60)
        overall = metrics['overall_score']
        print(f"  Overall Score: {overall['overall']:.2f}/100")
        print(f"  Grade: {overall['grade']}")
        print(f"  - Diversity: {overall['diversity_component']:.2f}")
        print(f"  - Quality: {overall['quality_component']:.2f}")
        print(f"  - Coherence: {overall['coherence_component']:.2f}")
        print("="*60 + "\n")


def evaluate_model_outputs(model, decode_fn, num_samples: int = 5,
                          max_tokens: int = 500, reference_corpus: str = None):
    """
    Generate multiple samples from a model and evaluate them.

    Args:
        model: The trained LLM model
        decode_fn: Function to decode token indices to text
        num_samples: Number of samples to generate
        max_tokens: Maximum tokens per sample
        reference_corpus: Optional reference text for comparison

    Returns:
        List of evaluation results for each sample
    """
    evaluator = LLMEvaluator(reference_corpus=reference_corpus)
    results = []

    print(f"\nGenerating and evaluating {num_samples} samples...\n")

    for i in range(num_samples):
        print(f"Sample {i+1}/{num_samples}")
        print("-" * 60)

        # Generate text
        context = torch.zeros((1, 1), dtype=torch.long, device=next(model.parameters()).device)
        generated_indices = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
        generated_text = decode_fn(generated_indices)

        print(f"Generated text preview:\n{generated_text[:200]}...\n")

        # Evaluate
        metrics = evaluator.evaluate(generated_text, verbose=True)
        results.append({
            'sample_id': i + 1,
            'text': generated_text,
            'metrics': metrics
        })

    # Aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS ACROSS ALL SAMPLES")
    print("="*60)

    avg_overall = np.mean([r['metrics']['overall_score']['overall'] for r in results])
    avg_diversity = np.mean([r['metrics']['diversity']['diversity_score'] for r in results])
    avg_repetition = np.mean([r['metrics']['diversity']['repetition_ratio'] for r in results])

    print(f"\nAverage Overall Score: {avg_overall:.2f}/100")
    print(f"Average Diversity Score: {avg_diversity:.4f}")
    print(f"Average Repetition Ratio: {avg_repetition:.4f}")

    return results


if __name__ == "__main__":
    # Example usage
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for evaluation.
    It contains multiple sentences with varying lengths. Some are short. Others are much longer
    and contain more complex structures with additional clauses and phrases.
    """

    evaluator = LLMEvaluator()
    metrics = evaluator.evaluate(sample_text)
