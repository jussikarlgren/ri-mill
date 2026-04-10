from datasets import load_dataset
data_sample = load_dataset("Eloquent/Voight-Kampff", "sample", split="train", download_mode="force_redownload")

data_2024 = load_dataset("Eloquent/Voight-Kampff", "test-2024", split="test")

data_2025 = load_dataset("Eloquent/Voight-Kampff", "test-2025", split="test")

data_2026 = load_dataset("Eloquent/Voight-Kampff", "test-2026", split="test")

print(data_2026)
