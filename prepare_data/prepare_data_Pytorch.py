import torch
from torch.utils.data import Dataset
import random
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.samples = self.generate_samples()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        sample = torch.tensor(self.samples[index], dtype=torch.float32)
        return sample[:-1], sample[-1] 

    def generate_samples(self):
        samples = []
        for _ in range(self.num_samples):
            a = random.uniform(-1, 1)
            b = random.uniform(-1, 1)
            c = random.uniform(-1, 1)
            label = 0 if a < 0 and b < 0 else 1
            samples.append([a, b, c, label])
        return samples

# Số lượng mẫu trong dataset
num_samples = 100

# Tạo dataset
custom_dataset = CustomDataset(num_samples)

# Dùng DataLoader để tạo iterable để sử dụng trong quá trình huấn luyện
data_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

