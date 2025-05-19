# test_dataset.py
import os
import numpy as np
import torch
from dataset_fast import MarioFastDataset, get_fast_loader

def create_dummy_episode(path, length=20):
    # Create dummy frames and actions
    frames = np.random.randint(0, 256, size=(length, 84, 84), dtype=np.uint8)
    actions = np.random.randint(0, 3, size=(length,), dtype=np.uint8)
    np.savez_compressed(path, frames=frames, actions=actions)


def test_dataset_length_and_shapes(tmp_path):
    # Prepare dummy episodes
    dir_path = tmp_path / "episodes"
    dir_path.mkdir()
    for i in range(3):
        create_dummy_episode(str(dir_path / f"ep_{i}.npz"), length=20)

    seq_len = 5
    ds = MarioFastDataset(str(dir_path), seq_len=seq_len)
    # Total samples: sum(length - seq_len) = 3*(20-5) = 45
    assert len(ds) == 3 * (20 - seq_len)

    # Test a few random indices
    for idx in [0, 10, 44]:
        x, y = ds[idx]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        # x shape: (seq_len, 1, 84, 84)
        assert x.shape == (seq_len, 1, 84, 84)
        # y is a scalar
        assert y.dim() == 0


def test_loader_batch_shapes(tmp_path):
    dir_path = tmp_path / "episodes2"
    dir_path.mkdir()
    for i in range(2):
        create_dummy_episode(str(dir_path / f"ep_{i}.npz"), length=10)

    loader = get_fast_loader(str(dir_path), batch_size=4, seq_len=3, num_workers=0)
    batch = next(iter(loader))
    x_batch, y_batch = batch
    assert x_batch.shape == (4, 3, 1, 84, 84)
    assert y_batch.shape == (4,)


# test_model.py
import torch
from model import MarioRNN

def test_model_forward_shape():
    batch_size, seq_len = 2, 7
    # Dummy input: random floats
    x = torch.rand(batch_size, seq_len, 1, 84, 84)
    model = MarioRNN(hidden_size=16, n_layers=1, n_actions=3)
    out = model(x)
    # Expected output shape: (batch_size, n_actions)
    assert out.shape == (batch_size, 3)
    # Values should be finite
    assert torch.isfinite(out).all()
