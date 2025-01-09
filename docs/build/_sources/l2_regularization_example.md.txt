# L2 Regularization

This example demonstrates how to use the BOAT library to perform bi-level optimization with L2 regularization. The example includes data preprocessing, model initialization, and the optimization process.

## Step 1: Configuration Loading

```python
with open(os.path.join(parent_folder, "configs/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(parent_folder, "configs/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)
```

### Explanation:
- **`boat_config_l2.json`**: Contains configuration for the bi-level optimization problem.
- **`loss_config_l2.json`**: Defines the loss functions for both upper-level and lower-level models.

---

## Step 2: Data Preparation

```python
trainset, valset, testset, tevalset = get_data(args)
torch.save(
    (trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pt")
)
```

### Explanation:
- The `get_data` function loads and splits the dataset into training, validation, testing, and evaluation sets.
- Processed data is saved to the specified `data_path` directory for future use.

---

## Step 3: Model Initialization

```python
device = torch.device("cpu")
n_feats = trainset[0].shape[-1]
upper_model = UpperModel(n_feats, device)
lower_model = LowerModel(n_feats, device, num_classes=trainset[1].unique().shape[-1])
```

### Explanation:
- **`UpperModel`**: Represents the upper-level model, optimizing high-level objectives.
- **`LowerModel`**: Represents the lower-level model, focusing on optimizing low-level objectives.

---

## Step 4: Optimizer Setup

```python
upper_opt = torch.optim.Adam(upper_model.parameters(), lr=0.01)
lower_opt = torch.optim.SGD(lower_model.parameters(), lr=0.01)
dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else []
hyper_method = args.hyper_method.split(",") if args.hyper_method else []
```

### Explanation:
- **Adam optimizer** is used for the upper-level model.
- **SGD optimizer** is applied to the lower-level model.
- The `dynamic_method` and `hyper_method` parameters allow for flexible optimization strategies.

---

## Step 5: Main Function

```python
b_optimizer = boat.Problem(boat_config, loss_config)
b_optimizer.build_ll_solver()
b_optimizer.build_ul_solver()

ul_feed_dict = {"data": trainset[0].to(device), "target": trainset[1].to(device)}
ll_feed_dict = {"data": valset[0].to(device), "target": valset[1].to(device)}
iterations = 30
for x_itr in range(iterations):
    b_optimizer.run_iter(
        ll_feed_dict, ul_feed_dict, current_iter=x_itr
    )
```

### Explanation:
- The `run_iter` function performs iterations of bi-level optimization using the BOAT library.
- Input feed dictionaries `ll_feed_dict` and `ul_feed_dict` are passed to define data and targets for lower-level and upper-level optimizations, respectively.

