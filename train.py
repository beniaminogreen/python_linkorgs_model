import csv
import os
import random
import tempfile
import time
import torch
import tqdm
import ray

import time

from hyperopt import hp
from model import LSTMEmbedding
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from torch import nn
from utils import n_letters, lineToTensor
from dataset import OrganizationDataset

from custom_loss import HardTripletLoss

dataset = OrganizationDataset()
x = dataset.random_set(10)

def train_embeddings(config):
    dataset = OrganizationDataset()

    model = LSTMEmbedding(
                 input_size = 40 ,
                 hidden_size = config['hidden_dimension'] ,
                 output_size= config['output_size'],
                 vocab_size = 256,
                 embedding_dim = config['embedding_dimension'],
                 bidirectional = False,
                 depth = config['depth']
                 )

    criterion = HardTripletLoss(.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    report_every = 200
    while True:
        total_loss = 0
        for i in range(report_every):
            training_triplets = dataset.random_set(1000)

            embeddings = [model(input) for input in training_triplets]

            for embedding in embeddings:
                assert len(embedding.size()) <= 2

            loss = criterion(embeddings[0], embeddings[1], embeddings[2])

            loss.backward()

            if config['clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.state_dict(),
                os.path.join(temp_checkpoint_dir, "model.pth")
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            report_dict = {
                    "loss": total_loss / report_every,
                    "bidirectional" : config['bidirectional'],
                    "output_size" : config['output_size'],
                    'hidden_dim': config['hidden_dimension'],
                    'embedding_dim': config['embedding_dimension'],
                    'lr' : config['learning_rate'],
                    'depth' : config['depth']
                    }
            train.report(report_dict, checkpoint=checkpoint)


context = ray.init()
print(context.dashboard_url)

search_space = {
    "learning_rate": hp.loguniform("learning_rate", -10, -1),
    "hidden_dimension" : hp.choice("hidden_dimension", [2**i for i in range(5,9)]),
    "embedding_dimension" : hp.choice("embedding_dimension", [2**i for i in range(5,9)]),
    "output_size" : hp.choice("output_size", [2**i for i in range(5,9)]),
    "clip" : hp.choice("clip", [True, False]),
    "bidirectional" : hp.choice("bidirectional", [True, False]),
    "depth" : hp.choice("depth", [1,2,3])
}



hyperopt_search = HyperOptSearch(search_space, metric="loss", mode="min")

tuner = tune.Tuner(
    train_embeddings,
    tune_config=tune.TuneConfig(
        num_samples=500,
        scheduler=ASHAScheduler(metric="loss", mode="min", max_t=300),
        search_alg=hyperopt_search,
        time_budget_s=6.5*60*60,
        max_concurrent_trials=11
    )
)

results = tuner.fit()

time.sleep(60*5)

best_trial_dir = results.get_best_result(metric="loss", mode="min")





