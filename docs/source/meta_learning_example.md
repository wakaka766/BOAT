# Meta-Learning

This example demonstrates how to use the BOAT library to perform meta-learning tasks, focusing on bi-level optimization using sinusoid functions as the dataset. The explanation is broken down into steps with corresponding code snippets.

---

## Step 1: Importing Libraries and Dependencies

```python
import os
import torch
import boat
from torch import nn
from torchmeta.toy.helpers import sinusoid
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm
from examples.meta_learning.util_ml import get_sinuoid
```

### Explanation:
- Import necessary libraries, including `torch`, `boat`, and `torchmeta`.

---

## Step 2: Dataset Preparation

```python
batch_size = 4
kwargs = {"num_workers": 1, "pin_memory": True}
device = torch.device("cpu")
dataset = sinusoid(shots=10, test_shots=100, seed=0)
```

### Explanation:
- **Dataset**: The `sinusoid` function generates toy sinusoidal data for meta-learning.
- **`batch_size`**: Number of tasks in each batch.
- **`device`**: Specify the computation device (CPU in this case).

---

## Step 3: Model and Optimizer Setup

```python
meta_model = get_sinuoid()
dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
test_dataloader = BatchMetaDataLoader(dataset, batch_size=batch_size, **kwargs)
inner_opt = torch.optim.SGD(lr=0.1, params=meta_model.parameters())
outer_opt = torch.optim.Adam(meta_model.parameters(), lr=0.01)
y_lr_schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=outer_opt, T_max=80000, eta_min=0.001
)
```

### Explanation:
- **Meta-Model**: Obtain a sinusoid-based meta-model using `get_sinuoid`.
- **DataLoader**: `BatchMetaDataLoader` creates meta-dataset loaders for training and testing.
- **Optimizers**: SGD for inner-loop optimization, Adam for outer-loop optimization.
- **Learning Rate Scheduler**: Gradually adjusts learning rates during training.

---

## Step 4: Configuration Loading

```python
base_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(base_folder)
with open(os.path.join(parent_folder, "configs/boat_config_ml.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "configs/loss_config_ml.json"), "r") as f:
    loss_config = json.load(f)
```

### Explanation:
- Load configurations for BOAT and loss functions from JSON files.

---

## Step 5: Bi-Level Optimization Setup

```python
dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else None
hyper_method = args.hyper_method.split(",") if args.hyper_method else None
boat_config["dynamic_op"] = dynamic_method
boat_config["hyper_op"] = hyper_method
boat_config["lower_level_model"] = meta_model
boat_config["upper_level_model"] = meta_model
boat_config["lower_level_var"] = list(meta_model.parameters())
boat_config["upper_level_var"] = list(meta_model.parameters())
boat_config["lower_level_opt"] = inner_opt
boat_config["upper_level_opt"] = outer_opt
b_optimizer = boat.Problem(boat_config, loss_config)
b_optimizer.build_ll_solver()
b_optimizer.build_ul_solver()
```

### Explanation:
- Configure and initialize the bi-level optimizer using BOAT.
- Define models, variables, and optimizers for both levels.

---

## Step 6: Main Function

```python
with tqdm(dataloader, total=1, desc="Meta Training Phase") as pbar:
    for meta_iter, batch in enumerate(pbar):
        ul_feed_dict = [
            {
                "data": batch["test"][0][k].float().to(device),
                "target": batch["test"][1][k].float().to(device),
            }
            for k in range(batch_size)
        ]
        ll_feed_dict = [
            {
                "data": batch["train"][0][k].float().to(device),
                "target": batch["train"][1][k].float().to(device),
            }
            for k in range(batch_size)
        ]
        loss, run_time = b_optimizer.run_iter(
            ll_feed_dict, ul_feed_dict, current_iter=meta_iter
        )
        y_lr_schedular.step()
        print("validation loss:", loss[-1][-1])
        if meta_iter >= 1:
            break
```

### Explanation:
- Iterate through batches using `tqdm` for progress visualization.
- Prepare feed dictionaries for lower-level and upper-level optimizations.
- Call `run_iter` for bi-level optimization, followed by updating the learning rate scheduler.
