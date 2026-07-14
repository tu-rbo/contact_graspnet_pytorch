from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from contact_graspnet_pytorch.checkpoints import CheckpointIO
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def __call__(self, _points):
        grasps = torch.eye(4).repeat(1, 3, 1, 1)
        return {
            "pred_grasps_cam": grasps,
            "pred_scores": torch.tensor([[0.9, 0.2, 0.7]]),
            "pred_points": torch.tensor(
                [[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]]]
            ),
            "offset_pred": torch.zeros((1, 3)),
        }


def test_inference_helpers_import_without_renderer_dependency() -> None:
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'pyrender' or name.startswith('pyrender.'):
        raise AssertionError('pyrender was imported by inference helpers')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import contact_graspnet_pytorch.data
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def _estimator_for_predictions() -> GraspEstimator:
    estimator = object.__new__(GraspEstimator)
    estimator.device = torch.device("cpu")
    estimator.model = _DummyModel()
    estimator._num_input_points = 3
    estimator._contact_grasp_cfg = {
        "DATA": {"gripper_width": 0.08, "num_point": 3, "raw_num_points": 3},
        "TEST": {
            "extra_opening": 0.0,
            "max_farthest_points": 2,
            "num_samples": 2,
            "first_thres": 0.5,
            "second_thres": 0.5,
            "with_replacement": False,
        },
    }
    estimator.select_grasps = lambda *_args, **_kwargs: np.array([0, 2])
    return estimator


def test_checkpoint_map_location_is_forwarded(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    calls = {}

    def fake_load(filename, map_location=None):
        calls["filename"] = filename
        calls["map_location"] = map_location
        return {}

    monkeypatch.setattr(torch, "load", fake_load)
    loader = CheckpointIO(checkpoint_dir=tmp_path, map_location="cpu")
    loader.load("model.pt")

    assert calls == {"filename": str(checkpoint), "map_location": "cpu"}


def test_force_cpu_overrides_cuda(monkeypatch) -> None:
    dummy_model = _DummyModel()
    model_module = SimpleNamespace(ContactGraspnet=lambda _cfg, _device: dummy_model)
    monkeypatch.setattr(
        "contact_graspnet_pytorch.contact_grasp_estimator.importlib.import_module",
        lambda _name: model_module,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    config = {"MODEL": {"model": "dummy"}, "DATA": {"raw_num_points": 3}}
    estimator = GraspEstimator(config, force_cpu=True)

    assert estimator.device == torch.device("cpu")
    assert dummy_model.device == torch.device("cpu")


def test_default_api_selects_grasps_and_return_all_is_opt_in(monkeypatch) -> None:
    estimator = _estimator_for_predictions()
    monkeypatch.setattr(
        "contact_graspnet_pytorch.contact_grasp_estimator.preprocess_pc_for_inference",
        lambda points, *_args, **_kwargs: (points, np.zeros((1, 3))),
    )
    points = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.0, 1.0]],
        dtype=np.float32,
    )

    selected = estimator.predict_grasps(points, convert_cam_coords=False)
    raw = estimator.predict_grasps(
        points,
        convert_cam_coords=False,
        return_all=True,
    )

    assert len(selected) == 4
    assert selected[0].shape == (2, 4, 4)
    assert selected[1].tolist() == pytest.approx([0.9, 0.7])
    assert len(raw) == 5
    assert raw[0].shape == (3, 4, 4)
    assert raw[4].tolist() == [0, 2]


def test_raw_scene_output_uses_keyed_selection_indices(monkeypatch) -> None:
    estimator = _estimator_for_predictions()
    monkeypatch.setattr(
        "contact_graspnet_pytorch.contact_grasp_estimator.regularize_pc_point_count",
        lambda points, _count: points,
    )
    monkeypatch.setattr(
        estimator,
        "predict_grasps",
        lambda *_args, **_kwargs: (
            np.zeros((3, 4, 4)),
            np.zeros(3),
            np.zeros((3, 3)),
            np.zeros(3),
            np.array([0, 2]),
        ),
    )

    outputs = estimator.predict_scene_grasps(
        np.zeros((3, 3)),
        return_all=True,
    )

    assert len(outputs) == 5
    assert outputs[4][-1].tolist() == [0, 2]


def test_raw_output_rejects_segment_filtering() -> None:
    estimator = _estimator_for_predictions()
    with pytest.raises(ValueError, match="incompatible"):
        estimator.predict_scene_grasps(
            np.zeros((3, 3)),
            filter_grasps=True,
            return_all=True,
        )
