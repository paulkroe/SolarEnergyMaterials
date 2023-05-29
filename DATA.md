GraphPrediction Task
-> ogbGraphormerDataset
    -> OGBDatasetLookupTable.GetOGBDataset
        -> MyPygGraphPropPredDataset -> MyPygPCQM4Mv2Dataset -> MyPygPCQM4MDataset
        -> GraphormerPYGDataset

    -> MyPygGraphPropPredDataset
        -> PgyGraphPropPredDataset 
    
    -> MyPygPCQM4Mv2Dataset
        -> PygPCQM4Mv2Dataset

    -> MyPygPCQM4MDataset
        -> PygPCQM4MDataset

    (this is done to use torch.distribution in order to download only once when using more than one process)

        -> PgyGraphPropPredDataset
        
        -> PygPCQM4Mv2Dataset

        -> PygPCQM4MDataset

        (these handle download and are part of ogb library, in the original graphormer implementation this class is implemented twice to avoid circular inport)

    -> GraphormerPYGDataset (handels data in the end)

