from ptflops import get_model_complexity_info
from transformers import ViTForImageClassification
import models.juanchitocnn
import models.visiontransformer
import models.efficientnetb0

"""
Get the FLOPs of a model for reporting
"""
# select_model = 'EfficientNet' # ['EfficientNet', 'VisionTransformer', 'JuanchitoCNN']

# if select_model == 'EfficientNet':
#     project_model = models.efficientnetb0.ProjectEfficientNet(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, optimizer=optimizer, momentum=momentum, weight_decay=weight_decay)
#     print(f'Using model {select_model}')
# elif select_model == 'VisionTransformer':
#     project_model = models.visiontransformer.ProjectVisionTransformer(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, optimizer=optimizer, momentum=momentum, weight_decay=weight_decay)
#     print(f'Using model {select_model}')
# elif select_model == 'JuanchitoCNN':
#     project_model = models.juanchitocnn.ProjectJuanchitoCNN(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, optimizer=optimizer, momentum=momentum, weight_decay=weight_decay)
#     print(f'Using model {select_model}')
# else:
#     print('No valid model selected')

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
print(f"FLOPs: {flops/1e9:.2f} G") # 1 gigaflop = 1 billion flop