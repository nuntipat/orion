import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_cifar_datasets,
    get_fashionMnist_datasets,
    mae, 
    train_on_cifar,
    get_fashionMnist_datasets,
    train_on_fashionMnist
)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/resnet.yml")
# trainloader, testloader = get_cifar_datasets(data_dir="../data", batch_size=32)
trainloader, testloader = get_fashionMnist_datasets(data_dir="../data", batch_size=1)
net = models.ResNetF()
# Train model (optional)
device = "cuda" if torch.cuda.is_available() else "cpu"
# train_on_cifar(net, data_dir="../data", epochs=1, device=device)
train_on_fashionMnist(net, data_dir="../data", epochs=1, device=device)

# --- Prepare for FHE ---
# Estimate input ranges and compile model for FHE
print("\nFitting and compiling for FHE...")
example_batch, _ = next(iter(trainloader))
orion.fit(net, example_batch)  
input_level = orion.compile(net)

# Switch network to FHE mode
net.he()

# --- FHE evaluation ---
print("\nStarting FHE inference on full test set...")
correct_fhe = 0
total = 0
start_total = time.time()

for inputs, targets in testloader:
    # Encode + encrypt batch
    vec_ptxt = orion.encode(inputs, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    # Run inference in FHE
    start = time.time()
    out_ctxt = net(vec_ctxt)
    end = time.time()

    # Decrypt + decode
    out_ptxt = out_ctxt.decrypt()
    outputs = out_ptxt.decode()

    # Convert to tensor for accuracy calculation
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)

    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct_fhe += predicted.eq(targets).sum().item()

    print(f"Batch runtime: {end-start:.4f} secs")

end_total = time.time()
acc_fhe = 100. * correct_fhe / total
print(f"\nFHE accuracy: {acc_fhe:.2f}%")
print(f"Total FHE runtime: {end_total - start_total:.2f} secs")
