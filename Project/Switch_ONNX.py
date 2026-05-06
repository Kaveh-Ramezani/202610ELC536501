import onnx
from onnx.external_data_helper import load_external_data_for_model

split_onnx_path = "onnx/edgeHeg_2D_Float32.onnx"
standalone_onnx_path = "onnx/edgeHegClassifier_PTQ_Standalone.onnx"

print(f"Loading split model: {split_onnx_path}...")
model = onnx.load(split_onnx_path)

print("Pulling external .data back into the main graph...")
load_external_data_for_model(model, "onnx/")

# --- THE MAGIC FIX FOR EDGE IMPULSE ---
print("Forcing IR version downgrade for Edge Impulse compatibility...")
model.ir_version = 7  # Version 7 is globally accepted by all legacy and modern toolchains
# --------------------------------------

print("Saving standalone model...")
onnx.save_model(model, standalone_onnx_path, save_as_external_data=False)
print(f"\nSuccess! Upload '{standalone_onnx_path}' to Edge Impulse.")