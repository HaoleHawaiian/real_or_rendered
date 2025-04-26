from ptflops import get_model_complexity_info
from transformers import ViTForImageClassification

"""
Get the FLOPs of a model for reporting
"""

# model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-distilled-patch16-224', num_labels=self.num_classes).to(self.device)
model = ViTForImageClassification.from_pretrained('facebook/deit-tiny-distilled-patch16-224', num_labels=2)
input_shape = (3, 224, 224)

macs, params = get_model_complexity_info(
    model, 
    input_shape,
    as_strings=False,
    print_per_layer_stat=False
)

flops = macs * 2  # Convert MACs to FLOPs
print(f"FLOPs: {flops/1e9:.2f} G")  # Expected ~1.3 GFLOPs