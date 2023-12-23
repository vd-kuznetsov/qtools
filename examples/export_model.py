import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig

from .utils import model_pipeline


def export_model(cfg: DictConfig):
    # There were problems with converting the pytorch quantized model to ONNX
    model = model_pipeline(cfg, evaluate=True, quantize=False)
    state_dict = torch.load(cfg.model.name + "_no_quant.pth")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        cfg.model.name + ".onnx",
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["IMAGES"],
        output_names=["CLASS_PROBS"],
        dynamic_axes={"IMAGES": {0: "BATCH_SIZE"}, "CLASS_PROBS": {0: "BATCH_SIZE"}},
    )

    # Comparing ort and torch outputs
    original_embeddings = model(dummy_input).detach().numpy()
    ort_inputs = {
        "IMAGES": dummy_input.numpy(),
    }
    ort_session = ort.InferenceSession(cfg.model.name + ".onnx")
    onnx_embeddings = ort_session.run(None, ort_inputs)[0]

    assert np.allclose(original_embeddings, onnx_embeddings, atol=1e-5)


if __name__ == "__main__":
    raise RuntimeError("Use `python commands.py mode=export`")
