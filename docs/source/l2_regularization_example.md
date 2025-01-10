# L2 Regularization with Jittor

This example demonstrates how to use the BOAT library with the Jittor framework to perform bi-level optimization with L2 regularization. The example includes data preprocessing, model initialization, and the optimization process.

## Step-by-Step Explanation

---

## Step 1: Configuration Loading

```python
base_folder = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(base_folder, "configs_jit/boat_config_l2.json"), "r") as f:
    boat_config = json.load(f)

with open(os.path.join(base_folder, "configs_jit/loss_config_l2.json"), "r") as f:
    loss_config = json.load(f)
```

### Explanation:
- **`boat_config_l2.json`**: Contains configuration for the bi-level optimization problem.
- **`loss_config_l2.json`**: Defines the loss functions for both upper-level and lower-level models.

---

## Step 2: Data Preparation

```python
def get_data(args):
    # Load and process the data
    trainset, valset, testset, tevalset = get_data(args)

    # Save the data for future use
    jit.save(
        (trainset, valset, testset, tevalset), os.path.join(args.data_path, "l2reg.pkl")
    )
    print(f"[info] successfully generated data to {args.data_path}/l2reg.pkl")
```

### Explanation:
- The `get_data` function loads the dataset, processes it to Jittor tensors, and splits it into training, validation, test, and evaluation sets.
- Processed data is saved to a file for future use.

---

## Step 3: Model Initialization

```python
class UpperModel(jit.Module):
    def __init__(self, n_feats):
        self.x = jit.init.constant([n_feats], "float32", 0.0).clone()

    def execute(self):
        return self.x


class LowerModel(jit.Module):
    def __init__(self, n_feats, num_classes):
        self.y = jit.zeros([n_feats, num_classes])
        jit.init.kaiming_normal_(
            self.y, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def execute(self):
        return self.y

upper_model = UpperModel(trainset[0].shape[-1])
lower_model = LowerModel(trainset[0].shape[-1], int(trainset[1].max().item()) + 1)
```

### Explanation:
- **`UpperModel`**: Represents the upper-level model with a single learnable parameter.
- **`LowerModel`**: Represents the lower-level model initialized using the Kaiming initialization strategy.

---

## Step 4: Optimizer Setup

```python
upper_opt = jit.nn.Adam(upper_model.parameters(), lr=0.01)
lower_opt = jit.nn.SGD(lower_model.parameters(), lr=0.01)

dynamic_method = args.dynamic_method.split(",") if args.dynamic_method else []
hyper_method = args.hyper_method.split(",") if args.hyper_method else []
```

### Explanation:
- **Adam optimizer**: Used for the upper-level model to update its parameters.
- **SGD optimizer**: Applied to the lower-level model for efficient gradient updates.
- The `dynamic_method` and `hyper_method` parameters allow flexible optimization strategies.

---

## Step 5: Bi-Level Optimization

```python
boat_config["dynamic_op"] = dynamic_method
boat_config["hyper_op"] = hyper_method
boat_config["lower_level_model"] = lower_model
boat_config["upper_level_model"] = upper_model
boat_config["lower_level_opt"] = lower_opt
boat_config["upper_level_opt"] = upper_opt
boat_config["lower_level_var"] = list(lower_model.parameters())
boat_config["upper_level_var"] = list(upper_model.parameters())

b_optimizer = boat.Problem(boat_config, loss_config)
b_optimizer.build_ll_solver()
b_optimizer.build_ul_solver()

ul_feed_dict = {"data": trainset[0], "target": trainset[1]}
ll_feed_dict = {"data": valset[0], "target": valset[1]}

iterations = 3 if "DM" in dynamic_method and "GDA" in dynamic_method else 2
for x_itr in range(iterations):
    if "DM" in dynamic_method and "GDA" in dynamic_method:
        b_optimizer._ll_solver.gradient_instances[-1].strategy = "s" + str(
            x_itr % 3 + 1
        )
    loss, run_time = b_optimizer.run_iter(ll_feed_dict, ul_feed_dict, current_iter=x_itr)
```

### Explanation:
- Configures the `boat_config` with models, optimizers, and variables for both levels.
- The `run_iter` function iterates over the bi-level optimization process using BOAT.

---

## Step 6: Evaluation

```python
def evaluate(x, w, testset):
    with jit.no_grad():
        test_x, test_y = testset
        y = test_x @ x
        y_np = y.numpy()
        test_y_np = test_y.numpy() if isinstance(test_y, jit.Var) else test_y
        loss = jit.nn.cross_entropy_loss(y, jit.array(test_y_np)).item()
        predicted = y_np.argmax(axis=-1)
        acc = (predicted == test_y_np).sum() / len(test_y_np)
    return loss, acc

test_loss, test_acc = evaluate(lower_model(), upper_model(), testset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
```

### Explanation:
- The `evaluate` function calculates the model's loss and accuracy on the test dataset.
- Outputs the test performance metrics for monitoring optimization progress.

---

## How to Run

To execute the example, use the following command:

```bash
python your_script_name.py --data_path ./data --model_path ./save_l2reg --dynamic_method NGD --hyper_method RAD
