
import torch
from torch_geometric.data import InMemoryDataset

class GNNDataset(InMemoryDataset):
    def __init__(self, root, types='train', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif types == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']

    def _download(self):
        pass

    def process(self):
        pass

if __name__ == "__main__":
    dataset = GNNDataset(root='preprocessed_data/celegans')
    print(dataset[0])
    print(dataset[0].x)
    
