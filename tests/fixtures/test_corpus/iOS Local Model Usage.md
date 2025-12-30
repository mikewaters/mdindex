# iOS Local Model Usage

<https://github.com/argmaxinc/DiffusionKit>

<https://github.com/argmaxinc/WhisperKit>

<https://github.com/Anemll/Anemll>

<https://news.ycombinator.com/item?id=43879702>

Apple devices all come with a special chip, the Apple Neural Engine (ANE), that is ideal for this type of model. Let's see what it takes to get it running!

<https://stephenpanaro.com/blog/modernbert-on-apple-neural-engine>

take the official model from HuggingFace and use Apple's coremltools to convert it to CoreML format. This format is required to utilize the ANE hardware.

Conversion is straightforward:


from transformers import AutoModelForMaskedLM
import torch
import coremltools as ct
import numpy as np

class Model(torch.nn.Module):
    def \__init_\_(self):
        super(Model, self).\__init_\_()
        self.model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask).logits

model = Model().eval()
input_ids = torch.zeros((1, 1024), dtype=torch.int32)
mask = torch.ones_like(input_ids)
ct.convert(
    torch.jit.trace(model, (input_ids, mask)),
    inputs=\[
        ct.TensorType(name="input_ids",
                      shape=input_ids.shape,
                      dtype=np.int32),
        ct.TensorType(name="attention_mask",
                      shape=mask.shape,
                      dtype=np.int32,
                      default_value=mask.numpy()),
    \],
    outputs=\[ct.TensorType(name="logits")\],
    minimum_deployment_target=ct.target.macOS14,
).save(f"ModernBERT-base-hf.mlpackage")
        
We can open the resulting model in Xcode and run a benchmark to see how fast it is.

<https://stephenpanaro.com/blog/modernbert-on-apple-neural-engine>

take the official model from HuggingFace and use Apple's coremltools to convert it to CoreML format. This format is required to utilize the ANE hardware.

Conversion is straightforward:


from transformers import AutoModelForMaskedLM
import torch
import coremltools as ct
import numpy as np

class Model(torch.nn.Module):
    def \__init_\_(self):
        super(Model, self).\__init_\_()
        self.model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base"
        )
    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask).logits

model = Model().eval()
input_ids = torch.zeros((1, 1024), dtype=torch.int32)
mask = torch.ones_like(input_ids)
ct.convert(
    torch.jit.trace(model, (input_ids, mask)),
    inputs=\[
        ct.TensorType(name="input_ids",
                      shape=input_ids.shape,
                      dtype=np.int32),
        ct.TensorType(name="attention_mask",
                      shape=mask.shape,
                      dtype=np.int32,
                      default_value=mask.numpy()),
    \],
    outputs=\[ct.TensorType(name="logits")\],
    minimum_deployment_target=ct.target.macOS14,
).save(f"ModernBERT-base-hf.mlpackage")
        
We can open the resulting model in Xcode and run a benchmark to see how fast it is.

ONNX is horrible for anything that has variable input shapes and that is why nobody uses it for LLMs. It fundamentally is poorly designed for anything that doesn't take a fixed size image.

Not really. Apple software uses the neural engine all over the place, but rarely do others. Maybe this will change \[1\]
 There was a guy using it for live video transformations and it almost caused the phones to “melt”. \[2\]
\[1\] https://machinelearning.apple.com/research/neural-engine-tra...
\[2\] https://x.com/mattmireles/status/1916874296460456089

<https://news.ycombinator.com/item?id=43881294>

Visual ai

<https://github.com/vlm-run/n8n-nodes-vlmrun>