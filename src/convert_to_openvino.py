import os
import torch
import openvino as ov
import yaml
from SCNet.scnet.SCNet import SCNet

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    # Flatten config for SCNet init (adjust as needed)
    model_cfg = config.get('model', {})
    return model_cfg

def load_scnet_model(checkpoint_path: str, config_path: str):
    model_cfg = load_config(config_path)
    model = SCNet(**model_cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    # If checkpoint is a dict with 'state_dict', use that
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

def export_to_openvino(model, example_input, output_path="ov_model/model.xml", compress_fp16=False):
    # Use torch.jit.script for better control flow support
    try:
        traced = torch.jit.script(model)
        print("✅ Model scripted with torch.jit.script.")
    except Exception as e:
        print(f"❌ torch.jit.script failed: {e}\nFalling back to torch.jit.trace (may be less robust for dynamic input shapes)")
        traced = torch.jit.trace(model, example_input)
    ov_model = ov.convert_model(traced)
    ov.save_model(ov_model, output_path)
    print(f"✅ OpenVINO model saved to {output_path} (+ .bin).")

def run_inference_openvino(xml_path, input_tensor):
    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, device_name="CPU")
    infer_request = compiled.create_infer_request()
    # Get input name
    input_name = compiled.input(0).get_any_name()
    result = infer_request.infer({input_name: input_tensor})
    return result

if __name__ == "__main__":
    ckpt = "checkpoints/SCNet-large.th"
    config_path = "checkpoints/config.yaml"
    model = load_scnet_model(ckpt, config_path)

    # Example: stereo, 11s, 44100Hz (from config)
    expected_sr = 44100
    expected_channels = 2
    expected_len = 11 * expected_sr
    dummy_input = torch.randn(1, expected_channels, expected_len, dtype=torch.float32)
    # Print input shape info
    print(f"[INFO] Exporting with dummy input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
    export_to_openvino(model, dummy_input, output_path="ov_model/model.xml", compress_fp16=False)

    # Test inference (optional)
    # output = run_inference_openvino("ov_model/model.xml", dummy_input.numpy())
    # print("✅ Inference output shape:", output)
