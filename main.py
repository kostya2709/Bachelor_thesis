import os
from torch.utils.data import DataLoader
from trajdata import AgentBatch, UnifiedDataset

# See below for a list of already-supported datasets and splits.
dataset = UnifiedDataset(
    desired_data=["nusc_mini"],
    data_dirs={  # Remember to change this to match your filesystem!
        "nusc_mini": "~/datasets/nuScenes"
    },
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=dataset.get_collate_fn(),
    num_workers=os.cpu_count(), # This can be set to 0 for single-threaded loading, if desired.
)

batch: AgentBatch
for batch in dataloader:
    # Train/evaluate/etc.
    pass
