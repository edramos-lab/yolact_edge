import argparse
import torch
import sys
import os
import traceback

# Add yolact_edge to the path
sys.path.insert(0, './yolact_edge')
from yolact import Yolact
from data.config import set_cfg


class YolactEdgeONNXWrapper(torch.nn.Module):
    """Wrapper class to simplify YOLACT Edge model for ONNX export"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Set model to eval mode
        self.model.eval()
        
        # Forward pass with proper extras for inference
        with torch.no_grad():
            # Create proper extras dictionary for inference
            extras = {
                "backbone": "full",
                "interrupt": False,
                "keep_statistics": False,
                "moving_statistics": None
            }
            output = self.model(x, extras=extras)
            
        # Return the prediction outputs
        if 'pred_outs' in output:
            return output['pred_outs']
        else:
            return output


def main():
    parser = argparse.ArgumentParser(description='Export YOLACT Edge model to ONNX')
    parser.add_argument('--trained_model', type=str, required=True, help='Path to the trained model .pth file')
    parser.add_argument('--config', type=str, required=True, help='Config name, e.g., yolact_edge_resnet101_config')
    parser.add_argument('--output_path', type=str, required=True, help='Output ONNX file path')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for ONNX export')
    parser.add_argument('--drop_weights', default=None)
    parser.add_argument('--coco_transfer', action='store_true', help='Transfer weights from COCO model')
    parser.add_argument('--yolact_transfer', action='store_true', help='Transfer weights from base YOLACT model')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--start_iter', type=int, default=-1, help='Resume training start iteration')
    args = parser.parse_args()

    # Check if model file exists
    if not os.path.exists(args.trained_model):
        print(f"Error: Model file {args.trained_model} not found!")
        return

    # Set config
    set_cfg(args.config)

    # Load model
    print(f"Loading model from {args.trained_model}...")
    net = Yolact(training=False)  # Set to False for inference
    
    # Load weights
    net.load_weights(args.trained_model, args)
    net.eval()

    # Create wrapper for ONNX export
    wrapped_model = YolactEdgeONNXWrapper(net)

    # Get config for input size
    from data.config import cfg
    input_size = cfg.max_size
    
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, input_size, input_size)
    
    print(f"Exporting model to ONNX with input shape: {dummy_input.shape}")
    print(f"Config: {args.config}")
    print(f"Output path: {args.output_path}")

    # Export to ONNX
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            args.output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['loc', 'conf', 'mask', 'priors', 'proto'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'loc': {0: 'batch_size'},
                'conf': {0: 'batch_size'},
                'mask': {0: 'batch_size'},
                'priors': {0: 'batch_size'},
                'proto': {0: 'batch_size'}
            },
            verbose=True
        )
        print(f"✅ Model successfully exported to {args.output_path}")
        
        # Print model info
        print(f"\nModel Information:")
        print(f"- Input shape: {dummy_input.shape}")
        print(f"- Output names: loc, conf, mask, priors, proto")
        print(f"- ONNX opset version: 11")
        
    except Exception as e:
        print(f"❌ Error exporting model: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        return

if __name__ == '__main__':
    main() 