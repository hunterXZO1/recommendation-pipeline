from pathlib import Path

from src.data.movielens import load_movielens_ratings
from src.models.pytorch_trainer import PyTorchModelTrainer


DATA_PATH = Path("data/movielens/ratings.dat")
MODEL_PATH = Path("artifacts/challenger.pt")


data = load_movielens_ratings(DATA_PATH)

print(f"Loaded {len(data)} interactions.")
print(f"Positive interactions: {(data['reward'] > 0).sum()}")

trainer = PyTorchModelTrainer(
    embedding_dim=64,
    epochs=5,
    batch_size=256,
    learning_rate=0.001,
)

model = trainer.train(data)

trainer.save(
    model,
    MODEL_PATH,
)

print(f"Model saved to: {MODEL_PATH}")