import time
import math
import torch
import orion
import orion.models as models
from orion.core.utils import (
    get_mnist_datasets,
    get_fashionMnist_datasets,
    mae, 
    train_on_mnist,
    train_on_fashionMnist
)

# Set seed for reproducibility
torch.manual_seed(42)

# Initialize the Orion scheme, model, and data
scheme = orion.init_scheme("../configs/mlp.yml")
trainloader, testloader = get_fashionMnist_datasets(data_dir="../data", batch_size=32)
net = models.MLP()

# Train model (optional)
device = "cuda" if torch.cuda.is_available() else "cpu"
train_on_fashionMnist(net, data_dir="../data", epochs=1, device=device)

# Get a test batch to pass through our network
inp, _ = next(iter(testloader))
print(inp.shape)

# Run cleartext inference
net.eval()

# Prepare for FHE inference. 
# Certain polynomial activation functions require us to know the precise range
# of possible input values. We'll determine these ranges by aggregating
# statistics from the training set and applying a tolerance factor = margin.
orion.fit(net, inp[0:1], batch_size=128)
input_level = orion.compile(net)

clear_outputs = []
with torch.no_grad():
    for inp, _ in testloader:
        out_clear = net(inp)
        clear_outputs.append(out_clear.detach().cpu())
# TODO: print accuracy of clear text function

# Encode and encrypt the input vector 
net.he()  # Switch to FHE mode

matches = []
runtimes = []
for idx, (inp, _) in enumerate(testloader):
    # Encode & encrypt
    vec_ptxt = orion.encode(inp, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)

    # Run FHE inference
    start = time.time()
    out_ctxt = net(vec_ctxt)
    end = time.time()

    # Decrypt + decode
    out_ptxt = out_ctxt.decrypt()
    out_fhe = out_ptxt.decode().cpu()

    # Take argmax (predicted class) for both clear and FHE outputs
    pred_clear = clear_outputs[idx].argmax(dim=1)   # shape (batch,)
    pred_fhe = out_fhe.argmax(dim=1)

    # Compute match (fraction of examples in batch with same predicted label)
    match = float((pred_clear == pred_fhe).float().mean().item())
    runtime = end - start

    matches.append(match)
    runtimes.append(runtime)

    print(f"Batch {idx+1}/{len(clear_outputs)} - Match: {match:.3f}, Runtime: {runtime:.4f} sec, clear={int(pred_clear.item())}, fhe={int(pred_fhe.item())}")

# Summary
avg_acc = sum(matches) / len(matches)
avg_time = sum(runtimes) / len(runtimes)
print(f"\nAverage accuracy (matching predicted label) over {len(clear_outputs)} batches: {avg_acc:.4f}")
print(f"Average runtime per batch: {avg_time:.4f} sec")
# dist = mae(out_clear, out_fhe)
# print(f"\nMAE: {dist:.4f}")
# print(f"Precision: {-math.log2(dist):.4f}")
# print(f"Runtime: {end-start:.4f} secs.\n")
