import onnx
import torch.onnx
from onnx_tf.backend import prepare
from torch.autograd import Variable

from unet import UNet




# Load the trained model from file
trained_model = UNet(n_channels=3, n_classes=5)
trained_model.load_state_dict(torch.load('checkpoints_doudouseban/CP_epoch50.pth',map_location='cpu'))

# Export the trained model to ONNX
dummy_input = Variable(torch.randn(1, 3, 300, 300)) # one black and white 300 x 300 x 3 picture will be the input to the model
torch.onnx.export(trained_model, dummy_input, "output/unet.onnx",opset_version=11)#,opset_version=11





# Load the ONNX file
model = onnx.load('output/unet.onnx')

# Import the ONNX model to Tensorflow
tf_rep = prepare(model)



# Input nodes to the model
print('inputs:', tf_rep.inputs)

# Output nodes from the model
print('outputs:', tf_rep.outputs)

# All nodes in the model
print('tensor_dict:')
print(tf_rep.tensor_dict)

tf_rep.export_graph('output/unet.pb')