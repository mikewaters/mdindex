# Model Guardrails Notes

On inference parameter tweaking and its impact on alignment

<https://news.ycombinator.com/item?id=44828901>

> Can anyone explain to me why they've removed parameter controls for temperature and top-p in reasoning models, including gpt-5?

> It's because all forms of sampler settings destroy safety/alignment. That's why top_p/top_k are still used and not tfs, min_p, top_n sigma, etc, why temperature is locked to 0-2 arbitrary range, etc
>
> 
