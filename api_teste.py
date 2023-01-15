from source.app_dataset import OntarioDataset, BrasilDataset

dataset = OntarioDataset(granularity='daily', cache=True)

# dataset = BrasilDataset(granularity='years', cache=True)




dataset.common_windows(['NO'])

print(dataset.windows.shape)