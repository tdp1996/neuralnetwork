import random

def generate_dataset(num_samples):
    dataset = [generate_sample() for _ in range(num_samples)]
    return dataset

def generate_sample():
    a = random.uniform(-1, 1)
    b = random.uniform(-1, 1)
    c = random.uniform(-1, 1)

    label = 0 if a < 0 and b < 0 else 1

    return ([a, b, c], label)

