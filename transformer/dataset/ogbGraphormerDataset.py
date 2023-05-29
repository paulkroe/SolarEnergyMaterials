from transformer.dataset.obgDatasetLookupTable import OGBDatasetLookupTable
class ogbGraphDataset:
    def __init__(
            self,
            dataset_name,
            seed: int = 0
            ):
        super().__init__()
        self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_name, seed=seed) #TODO: implement
        self.train_ind = self.dataset.train_ind
        self.valid_ind = self.dataset.valid_ind
        self.test_ind = self.dataset.test_ind

        self.dataset_train = self.dataset.train_data
        self.dataset_valid = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data
        
