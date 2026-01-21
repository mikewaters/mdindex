# Mobile LLM apps

## Flutter

<https://github.com/BrutalCoding/aub.ai>

AubAI brings you on-device gen-AI capabilities, including offline text generation and more, directly within your app.



<https://news.ycombinator.com/item?id=39557098>

Thanks for asking: Yes I did make it, but, no app tying it all together. At least, it isn't out yet.  The grunt work of getting it running on different platforms + nice easy OpenAI compatible interfaces x RAG x voice assistant is open source:

- FLLAMA: <https://github.com/Telosnex/fllama> llama.cpp at core, openai compatible API, function call support, multimodal model support, Metal support. All platforms incl. web, but WASM is slow, def. not worth it except as a proof of concept.

- FONNX: <https://github.com/Telosnex/fonnx> ONNX runtime at core, all platforms including web. Whisper, Silero VAD, Magika, and two embeddings models. (Mini LM L6 V3 is best for RAG)
   EDIT: I knew I recognized your username! [Aub.ai](http://Aub.ai)! Cheers, what you did with [aub.ai](http://aub.ai) convinced me it was possible to do llama.cpp in flutter with a high bar for engineering quality. Other stuff seemed a tad rushed, unstable, and not complete. Also congrats, just saw your recent update, been hoping something good came through and it did.