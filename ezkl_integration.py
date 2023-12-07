# ezkl_integration.py

import ezkl
import os
import json
import torch
import numpy as np
from termcolor import colored

# Paths for ezkl files
model_path = "neutron_detection.onnx"
compiled_model_path = "neutron_detection.compiled"
pk_path = "pk.key"
vk_path = "vk.key"
settings_path = "settings.json"
srs_path = "kzg.srs"
witness_path = "witness.json"
proof_path = "model.proof"

# Load data and prepare input.json
sample_index = 142  # Sample index you want to test
data, labels = np.load("data.npy"), np.load("labels.npy")
input_data = torch.tensor(
    data[sample_index], dtype=torch.float32
).unsqueeze(1)
data_array = input_data.detach().numpy().reshape([-1]).tolist()
json.dump({"input_data": [data_array]}, open("input.json", "w"))

# Print colored console messages
print(colored("Starting EZKL Integration", "cyan"))

# EZKL setup
py_run_args = ezkl.PyRunArgs()
py_run_args.input_visibility = "public"
py_run_args.output_visibility = "public"
py_run_args.param_visibility = "fixed"

print(colored("Generating settings...", "yellow"))
res = ezkl.gen_settings(model_path, settings_path, py_run_args=py_run_args)
assert res == True

print(colored("Calibrating settings...", "yellow"))
res = ezkl.calibrate_settings("input.json", model_path, settings_path, "resources")
assert res == True

print(colored("Compiling circuit...", "yellow"))
res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
assert res == True

# srs path
res = ezkl.get_srs(srs_path, settings_path)

# Generate witness
print(colored("Generating witness...", "green"))
res = ezkl.gen_witness("input.json", compiled_model_path, witness_path)
assert os.path.isfile(witness_path)

print(colored("Setting up the circuit...", "yellow"))
res = ezkl.setup(compiled_model_path, vk_path, pk_path, srs_path)
assert res == True

# Prove
print(colored("Generating proof...", "green"))
res = ezkl.prove(
    witness_path, compiled_model_path, pk_path, proof_path, srs_path, "single"
)
assert os.path.isfile(proof_path)

# Verify
print(colored("Verifying proof...", "green"))
res = ezkl.verify(proof_path, settings_path, vk_path, srs_path)
assert res == True

print(colored("Proof verified successfully!", "blue"))
