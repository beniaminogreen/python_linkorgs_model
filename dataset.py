import csv
import random
import torch

from utils import n_letters, unicodeToAscii

def letterToIndex(letter):
    return(ord(letter) - 32)

def line_to_tensor(line):
    tensor = torch.zeros(len(line))
    for idx, letter in enumerate(line):
        tensor[idx] = letterToIndex(letter)
    return tensor.to(torch.int32)

class OrganizationDataset:
    def __init__(self):
        organization_dict = {}
        with open('/home/beniamino/programming/python_linkorgs_models/data/training_data.csv', mode='r') as infile:
            reader = csv.reader(infile)
            for line in reader:
                value, key = line
                value = unicodeToAscii(value).ljust(40)[:40]

                if key in organization_dict:
                    organization_dict[key].add(value)
                else :
                    organization_dict[key] = set([value])

        self.organizations = [list(org_names) for org_names in organization_dict.values()]
        self.organizations = [orgs for orgs in self.organizations if len(orgs) > 1]
        self.n_organizations = len(self.organizations)

    def random_organization(self):
        organization_index = random.randint(0,  self.n_organizations - 1)
        organization_name = random.choice(self.organizations[organization_index])

        return (organization_index, organization_name)

    def random_triplet(self):
        anchor_idx, anchor_name = self.random_organization()
        negative_idx, negative_name = self.random_organization()

        positive_name = anchor_name
        while anchor_name == positive_name:
            positive_name = random.choice(self.organizations[anchor_idx])

        return((line_to_tensor(anchor_name), line_to_tensor(positive_name), line_to_tensor(negative_name)))


    def random_triplets(self, n):
        triplets = [self.random_triplet() for _ in range(n)]

        anchors = torch.stack([triplet[0] for triplet in triplets])
        positives = torch.stack([triplet[1] for triplet in triplets])
        negatives = torch.stack([triplet[2] for triplet in triplets])

        return((anchors, positives, negatives))


    def random_set(self, n_negatives):
        anchor_idx, anchor_name = self.random_organization()

        anchor = torch.stack([line_to_tensor(anchor_name)])

        positives = torch.stack([line_to_tensor(name) for name in self.organizations[anchor_idx] if name != anchor_name])

        negatives = []
        i = 0
        while i < n_negatives:
            negative_idx, negative_name = self.random_organization()
            if negative_idx != anchor_idx:
                i += 1
                negatives.append(line_to_tensor(negative_name))

        negatives = torch.stack(negatives)

        return((anchor, positives, negatives))



