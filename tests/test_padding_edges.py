import importlib.util
import sys
import types
from pathlib import Path

import pytest


class _Size(tuple):
    def __new__(cls, values):
        return super().__new__(cls, values)


class _FakeView:
    def __init__(self):
        self.cloned = False

    def clone(self):
        self.cloned = True
        return ("cloned", self)


class _FakeBatch:
    def __init__(self, shape):
        self.shape = _Size(shape)
        self.view = _FakeView()
        self.last_index = None

    def __getitem__(self, index):
        self.last_index = index
        return self.view


def _load_padding_module(monkeypatch):
    torch_module = types.ModuleType("torch")
    torch_module.Size = _Size
    torch_module.Tensor = object

    torch_nn = types.ModuleType("torch.nn")
    torch_functional = types.ModuleType("torch.nn.functional")

    def unexpected_pad(*args, **kwargs):
        raise AssertionError("F.pad should not be called by these edge-case tests")

    torch_functional.pad = unexpected_pad
    torch_nn.functional = torch_functional

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", torch_functional)

    module_path = (
        Path(__file__).resolve().parents[1] / "mfai" / "pytorch" / "padding.py"
    )
    spec = importlib.util.spec_from_file_location(
        "mfai_padding_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_padding_helpers_split_odd_deltas_to_trailing_side(monkeypatch):
    padding = _load_padding_module(monkeypatch)

    assert padding._get_2D_padding(  # noqa: SLF001
        new_shape=_Size([8, 9]),
        old_shape=_Size([5, 4]),
    ) == (2, 3, 1, 2)
    assert padding._get_3D_padding(  # noqa: SLF001
        new_shape=_Size([7, 8, 9]),
        old_shape=_Size([4, 5, 4]),
    ) == (2, 3, 1, 2, 1, 2)


def test_pad_batch_returns_same_object_when_shape_already_matches(monkeypatch):
    padding = _load_padding_module(monkeypatch)
    batch = _FakeBatch([2, 3, 5, 7])

    assert padding.pad_batch(batch=batch, new_shape=_Size([5, 7])) is batch


def test_pad_batch_rejects_unsupported_new_shape_length(monkeypatch):
    padding = _load_padding_module(monkeypatch)

    with pytest.raises(
        ValueError, match="new_shape must be a torch.Size of length 2 or 3"
    ):
        padding.pad_batch(batch=_FakeBatch([2, 3, 5, 7]), new_shape=_Size([8]))


def test_undo_padding_rejects_unsupported_old_shape_length(monkeypatch):
    padding = _load_padding_module(monkeypatch)

    with pytest.raises(
        ValueError, match="old_shape must be a torch.Size of length 2 or 3"
    ):
        padding.undo_padding(batch=_FakeBatch([2, 3, 5, 7]), old_shape=_Size([3]))


def test_undo_padding_clones_by_default(monkeypatch):
    padding = _load_padding_module(monkeypatch)
    batch = _FakeBatch([1, 1, 4, 6])

    result = padding.undo_padding(batch=batch, old_shape=_Size([2, 4]))

    assert result == ("cloned", batch.view)
    assert batch.view.cloned is True
    assert batch.last_index == (
        Ellipsis,
        slice(1, 3, None),
        slice(1, 5, None),
    )
