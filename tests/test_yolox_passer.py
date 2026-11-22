import pytest
import torch

from src.inference.passer import parse_yolox


def test_yolox_scores_letterbox_and_class_aware_nms():
    output = torch.tensor([[
        [208, 104, 104, 104, 0.9, 0.8, 0.1],
        [208, 104, 104, 104, 0.8, 0.8, 0.1],
        [208, 104, 104, 104, 0.9, 0.1, 0.9],
        [208, 300, 100, 100, 0.9, 0.9, 0.1],
        [0, 0, 5, 5, float('nan'), 0.9, 0.1],
    ]])
    original = output.clone()
    detections = parse_yolox(output, (200, 400), labels=('head', 'helmet'))
    assert len(detections) == 2
    assert {d.label for d in detections} == {'head', 'helmet'}
    assert all(d.box == (150, 50, 250, 150) for d in detections)
    assert torch.allclose(output, original, equal_nan=True)


def test_yolox_empty_and_invalid_shapes():
    assert parse_yolox(torch.empty((1, 0, 6)), (720, 1280)) == []
    with pytest.raises(ValueError):
        parse_yolox(torch.empty((2, 10, 6)), (720, 1280))
    with pytest.raises(ValueError):
        parse_yolox(torch.empty((1, 10, 7)), (720, 1280))
