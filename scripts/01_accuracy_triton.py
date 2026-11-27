import sys
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import triton_to_np_dtype

# 1. Model name from the command line e.g. resnet0_fp16
model = sys.argv[1]
client = httpclient.InferenceServerClient("localhost:8000")

# 2. Preprocessed validation data (from 09_prep_val.py)
images = np.load("data/val_images.npy")
labels = np.load("data/val_labels.npy")

# 3. Input dtype from the model metadata (the API contract)
meta = client.get_model_metadata(model)
dtype = triton_to_np_dtype(meta["inputs"][0]["datatype"])
print("Model input dtype:", meta["inputs"][0]["datatype"])

# 4. Predict in batches of 32 and count correct answers
correct = 0
for i in range(0, len(images), 32):
	batch = images[i:i+32].astype(dtype)
	inp = httpclient.InferInput("input", batch.shape, meta["inputs"][0]["datatype"])
	inp.set_data_from_numpy(batch)
	logits = client.infer(model, [inp]).as_numpy("logits")
	correct += (logits.argmax(axis=1) == labels[i:i+32]).sum()
print(f"Correct: {correct} / {len(images)}")
print(f"Accuracy: {100* correct / len(images)}%")
