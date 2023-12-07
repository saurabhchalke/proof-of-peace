import torch
from model import NeutronDetectorCNN


def export_onnx_model(model_file, onnx_file, image_size):
    model = NeutronDetectorCNN(image_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    dummy_input = torch.randn(1, 1, image_size, image_size, requires_grad=True)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    export_onnx_model("neutron_detector.pth", "neutron_detection.onnx", image_size=10)
