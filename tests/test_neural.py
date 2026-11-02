import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_saved_segmenter_architecture_reloads_without_network(tmp_path, monkeypatch):
    from transformers import SegformerConfig
    from recovery.models.nets import NeuralSegmenter, build_segmenter
    # Deliberately differs from the default B0 architecture that old loads used.
    config = SegformerConfig(num_labels=1, num_channels=7, depths=[1, 1, 1, 1],
                              hidden_sizes=[8, 16, 20, 32], num_attention_heads=[1, 2, 5, 8],
                              decoder_hidden_size=16).to_dict()
    net = build_segmenter(config=config)
    torch.save(net.state_dict(), tmp_path / "seg.pt")
    (tmp_path / "config.json").write_text(json.dumps(dict(n_frames=2, segmenter_config=config)))
    def no_network(*args, **kwargs):
        raise AssertionError("checkpoint restoration must use saved metadata")
    monkeypatch.setattr(SegformerConfig, "from_pretrained", no_network)
    restored = NeuralSegmenter(tmp_path, dev="cpu")
    assert restored.net.config.hidden_sizes == [8, 16, 20, 32]
    for key, value in net.state_dict().items():
        assert torch.equal(value, restored.net.state_dict()[key])
    result = restored.measure(np.zeros((512, 512, 3), np.uint8))
    assert result.is_valid


def test_random_segmenter_uses_selected_architecture_metadata(monkeypatch):
    from transformers import SegformerConfig
    from recovery.models.nets import build_segmenter
    calls = []
    def config_from_repo(name, **kw):
        calls.append(name)
        return SegformerConfig(depths=[1, 1, 1, 1], hidden_sizes=[8, 16, 20, 32], **kw)
    monkeypatch.setattr(SegformerConfig, "from_pretrained", config_from_repo)
    net = build_segmenter("example/custom-mit")
    assert calls == ["example/custom-mit"]
    assert net.config.hidden_sizes == [8, 16, 20, 32]
