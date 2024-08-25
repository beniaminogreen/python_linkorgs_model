import onnxruntime
import numpy as np
import tqdm

from model import LSTMEmbedding
from utils import n_letters, lineToTensor
from dataset import OrganizationDataset

ort_session = onnxruntime.InferenceSession("lstm_embedding.onnx", providers=['CPUExecutionProvider'])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

dataset = OrganizationDataset()


def run_through_model(tensor):
    modelInput = tensor.unsqueeze(0)
    out  = ort_session.run(None, {'modelInput' : to_numpy(modelInput)})

    return(out[0][0, :, :])


print(f"pos_dist, neg_dist")
for _ in tqdm.tqdm(range(10000)):
    random_triplet = dataset.random_triplet()
    anchor = random_triplet[0]
    positive = random_triplet[1]
    negative = random_triplet[2]

    anch_em = run_through_model(anchor)
    pos_em = run_through_model(positive)
    neg_em = run_through_model(negative)

    dist_pos = np.linalg.norm(anch_em-pos_em)
    dist_neg = np.linalg.norm(anch_em-neg_em)

    print(f"{dist_pos:.2f}, {dist_neg:.2f}")
