# Pre-training, fine-tuning, or in-context learning

> They, in my experience, result in different outcomes. **Fine-tuning is better at shaping the response, for example getting a correct JSON format,** or making the responses shorters and concise, or use emojis, or whatever. **Long prompts are better for in-context learning**, **fine-tuning doesn't seem to impart knowledge well**, **it just increases or decreases the likelihoods of existing knowledge coming out.**

> > I would have expected fine-tuning to be good at imparting knowledge. **Pre-training is often done for a single epoch only and models soak up the knowledge like crazy without multiple passes so why would fine-tuning be any different?**

Because **the learning rate vs. pre-training is completely different.** This is not accurate, but my mental model is that **the LLM's initial training establishes the "space of concepts and ideas" while tuning (like RLHF and fine-tuning) changes how it expresses those concepts and ideas.** It works well for me in deciding my approach.

## Fine tuning a smaller model

For some task, you can prompt an LLM to generate request/response (input/output) pairs to use for fine-tuning an SLM that is optimized for that task.

> This is partially correct, fine-tuning can be a step of this process. The full pipeline looks more like:
>
> 1\. Create a prompt which gets a large model to output (mostly) correct responses. 2. Build a dataset of those inputs/outputs, probably with human or LLM curation/judging in the loop since some will still be wrong. 3. Fine tune a small model on those inputs/outputs.
>
> **Now you have a smaller model which behaves more like the large model you were able to prompt engineer into instruction following.**

source: <https://news.ycombinator.com/item?id=41302234>

## Notes

Isn’t there a risk of over fitting here?

How is this related to LoRa?

<https://github.com/hiyouga/LLaMA-Factory>