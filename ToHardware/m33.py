import os
import onnx
import numpy as np
import onnxruntime.quantization as quant

# ==========================================
# 1. Calibration Data Reader
# ==========================================
class CalibrationDataReader(quant.CalibrationDataReader):
    """
    Reads a subset of typical input data (e.g., 100 samples) to figure out 
    the min/max ranges of the float32 tensors. This ensures accurate int8 scaling.
    """
    def __init__(self, calibration_data_path: str, input_name: str):
        print(f"Loading calibration data from: {calibration_data_path}")
        # Load your representative dataset (e.g., a .npy file with shape [100, Channels, Height, Width])
        self.data = np.load(calibration_data_path).astype(np.float32)
        self.input_name = input_name
        self.enum_data = iter(self.data)

    def get_next(self):
        next_data = next(self.enum_data, None)
        if next_data is not None:
            # CMSIS-NN usually expects a batch size of 1 for inference
            # Expand dimensions to make it [1, C, H, W]
            return {self.input_name: np.expand_dims(next_data, axis=0)}
        return None

# ==========================================
# 2. The Quantization Pipeline
# ==========================================
def optimize_for_cortex_m(input_model_path: str, output_model_path: str, calibration_data_path: str):
    """
    Takes a float32 ONNX model, quantizes it to int8 for ARM DSP instructions, 
    and saves the optimized model with embedded scales and zero-points.
    """
    print(f"Loading original float32 model: {input_model_path}")
    
    # Load the model to find the name of the input tensor automatically
    model = onnx.load(input_model_path)
    # Get the name of the first input node
    input_name = model.graph.input[0].name 
    print(f"Detected model input name: '{input_name}'")

    # Initialize the data reader
    data_reader = CalibrationDataReader(calibration_data_path, input_name)

    print("Running static Post-Training Quantization (PTQ)...")
    # Execute the ONNX Runtime quantizer
    quant.quantize_static(
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=data_reader,
        quant_format=quant.QuantFormat.QOperator, # Required for best CMSIS-NN compatibility
        activation_type=quant.QuantType.QInt8,    # 8-bit activations
        weight_type=quant.QuantType.QInt8,        # 8-bit weights
        optimize_model=True                       # Fold constants and remove redundant nodes
    )
    
    print(f"Success! INT8 optimized model saved to: {output_model_path}")
    return output_model_path

# ==========================================
# 3. The C-Code Generator Stub
# ==========================================
def generate_c_code(int8_model_path: str, output_dir: str):
    """
    This is where your Jinja2 templates will take over.
    It reads the newly created INT8 model, extracts the arrays, scales, 
    and zero-points, and writes them to model.c and model.h
    """
    print(f"\n--- Starting C Code Generation ---")
    print(f"Parsing optimized graph from {int8_model_path}...")
    
    # Load the INT8 model
    optimized_model = onnx.load(int8_model_path)
    
    # TODO: 1. Loop through optimized_model.graph.node
    # TODO: 2. Extract weights, zero-points, and scales
    # TODO: 3. Transpose weights from CHW to HWC for CMSIS-NN
    # TODO: 4. Pass variables to Jinja2 template and render C files
    
    print("C code generation complete (Placeholder).")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Define file paths
    FLOAT32_MODEL = "my_original_model.onnx"
    INT8_MODEL = "my_optimized_model.onnx"
    CALIB_DATA = "calibration_samples.npy"
    
    # Setup dummy files for testing the script logic (Remove this block in production)
    if not os.path.exists(CALIB_DATA):
        print("Generating dummy calibration data for testing...")
        # Simulating 50 images of 1x28x28 (e.g., MNIST)
        dummy_data = np.random.rand(50, 1, 28, 28).astype(np.float32) 
        np.save(CALIB_DATA, dummy_data)
        
    if not os.path.exists(FLOAT32_MODEL):
        print(f"WARNING: Please place a valid ONNX model named '{FLOAT32_MODEL}' in this directory.")
    else:
        # 1. Quantize the model
        optimized_path = optimize_for_cortex_m(FLOAT32_MODEL, INT8_MODEL, CALIB_DATA)
        
        # 2. Pass the new model to the C-generator
        generate_c_code(optimized_path, "./output_src")