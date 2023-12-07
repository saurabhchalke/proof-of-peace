import onnxruntime as ort
import torch
import numpy as np

# Define the device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def run_onnx_model(onnx_file, input_data):
    ort_session = ort.InferenceSession(onnx_file)
    outputs = ort_session.run(
        None, {"input": input_data.cpu().numpy()}
    )  # Move to CPU before conversion
    return outputs[0]


if __name__ == "__main__":
    # Load the model
    onnx_file = "neutron_detection.onnx"
    sample_data = np.load("data.npy")

    # Test with real data (should output positive for nuke)
    real_data = (
        torch.tensor(sample_data[142:143], dtype=torch.float32).unsqueeze(1).to(device)
    )
    real_output = run_onnx_model(onnx_file, real_data)
    print(f"Real Data Output: {real_output[0]} (Positive for Nuke)")

    # Test with random invalid data (should output negative)
    invalid_data = torch.rand(1, 1, 10, 10).to(device)  # Create random invalid data
    invalid_output = run_onnx_model(onnx_file, invalid_data)
    print(f"Invalid Data Output: {invalid_output[0]} (Negative)")
