# from typing import Optional

# from pathlib import Path

# import torch
# from torch.utils.data import Dataset
# import numpy as np

# class CatalanJuvenileJustice(Dataset):
    
#     def __init__(
#         self,
#         data_path: str,
#     ):
#         # Create as Path-object
#         self.data_path = Path(data_path)

#         # Load processed data-file
#         datafile = torch.load(self.data_path)

#         # Extract data
#         self.columns = datafile['data']['columns']
#         self.data = torch.FloatTensor(datafile['data']['content'])

#         self.sensitive_attributes = datafile['sensitive_attributes']['name']
#         self.sensitive_data = torch.FloatTensor(datafile['sensitive_attributes']['content'])

#         # Extract labels
#         self.target_name = datafile['labels']['name']
#         self.labels = torch.LongTensor(datafile['labels']['content'])

#         # Define number of classes and points
#         self.n_classes = self.labels.unique().__len__()
#         self.n_points, self.n_attributes = self.data.shape

#     def getColumns(self):
#         return self.columns

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         return {
#             "data": self.data[item, :], 
#             "label": self.labels[item], 
#             "sensitive_data": self.sensitive_data[item, :]
#         }

#     def get_loaders(
#         self,
#         batch_size: int,
#         shuffle: bool,
#         num_workers: int = 1,
#         test_size: float = 0.2,
#         val_size: float = 0.2,
#         split_type: Optional[str] = None,
#     ):

#         if split_type is None:
#             return torch.utils.data.DataLoader(
#                 self,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 num_workers=num_workers,
#             )

#         elif split_type == 'random':
#             train_size = np.round((1 - test_size) * (1 - val_size), 5)
#             val_size = np.round((1 - test_size) * val_size, 5)
#             assert np.allclose(train_size + val_size, 1 - test_size), "Proportions of the random split does not add up..."

#             train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(self, [train_size, val_size, test_size])
#             return [
#                 torch.utils.data.DataLoader(
#                     dataset_,
#                     batch_size=batch_size,
#                     shuffle=shuffle,
#                     num_workers=num_workers,
#                 )
#                 for dataset_ in [train_dataset, val_dataset, test_dataset]
#             ]
