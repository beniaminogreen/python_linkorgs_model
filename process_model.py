import torch
import onnx


from model import LSTMEmbedding
from utils import n_letters, lineToTensor
from dataset import OrganizationDataset

model_path = "model.pth"

model = LSTMEmbedding(
                 input_size = 40 ,
                 hidden_size = 32,
                 output_size= 32,
                 vocab_size = 256,
                 embedding_dim = 128,
                 bidirectional = False,
                 depth = 2
        )
model.load_state_dict(torch.load(model_path))

dataset = OrganizationDataset()

test_input = dataset.random_triplet()[0]
test_input = test_input.unsqueeze(0)

model(test_input)


torch.onnx.export(model = model,
         args = test_input,
         f="lstm_embedding.onnx",
         export_params=True,
         do_constant_folding=True,
         input_names = ['input'],
         output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}}
         )

onnx_model = onnx.load("lstm_embedding.onnx")
print(onnx.checker.check_model(onnx_model))
