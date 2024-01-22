from prepare_data.prepare_data import generate_sample, generate_dataset

def test_generate_sample():
    sample = generate_sample()

    assert len(sample) == 2 
    assert isinstance(sample, tuple)
    assert sample[1]==1 or sample[1]==0


def test_generate_dataset():
    number_sample = 100
    dataset = generate_dataset(number_sample)

    assert len(dataset) == 100
    assert isinstance(dataset, list)