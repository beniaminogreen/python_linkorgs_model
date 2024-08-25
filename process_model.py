import torch

from model import LSTMEmbedding
from utils import n_letters, lineToTensor
from dataset import OrganizationDataset

model_path = "/home/beniamino/ray_results/train_embeddings_2024-08-19_19-04-14/train_embeddings_7584abea_70_clip=False,depth=1,embedding_dimension=128,learning_rate=0.0009_2024-08-19_20-07-59/checkpoint_000299/model.pth"

model = LSTMEmbedding(n_letters, 128, 1)

dataset = OrganizationDataset()

test_input = dataset.random_triplet()[0]
test_input = torch.unsqueeze(test_input, 0)

model.load_state_dict(torch.load(model_path))

torch.onnx.export(model = model,
         args = test_input,
         f="lstm_embedding.onnx",
         export_params=True,
         do_constant_folding=True,
         input_names = ['modelInput'],
         output_names = ['modelOutput'],
         )
