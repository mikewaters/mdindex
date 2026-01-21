---
tags:
  - document 📑
  - landscape
---
# Uncensored Generative AI

Terms: Worm GPT, from a darknet gpt

# LLMs

Several uncensored large language models (LLMs) are available to the public, either for free or through paid services. These models are often open-source, allowing users to access and modify them with minimal restrictions. Here are some notable examples:

## **Free and Open-Source LLMs**

1. **PygmalionAI**: This model is a refined version of Meta’s LLaMA-13B, fine-tuned with data that includes NSFW content. It is available for free public use through platforms like Hugging Face\[1\].

2. **BLOOM**: Available through the Hugging Face ecosystem, BLOOM is an open-source LLM that can be used for free. It provides both pre-trained models and source code to the public\[2\].

3. **LLaMA 2**: Developed by Meta, LLaMA 2 is a popular open-source LLM that has been widely adopted for various applications. It is part of a broader movement to make LLMs more accessible and transparent\[3\].

4. **MPT-7b, Falcon, and StableLM**: These models have been part of the open-source LLM landscape, providing diverse capabilities for different applications\[3\].

## **Paid and Hosted LLMs**

While many open-source models are free, some hosted services may charge for access or usage. These services often provide additional features, such as scalable infrastructure or enhanced performance, which can be beneficial for commercial applications.

## **Considerations for Using Uncensored LLMs**

- **Safety and Ethics**: Uncensored models may lack safety checks, making them susceptible to misuse. It's important for users to implement their own safeguards to prevent harmful applications\[1\].

- **Customization and Control**: Open-source models allow for greater customization and control, enabling users to tailor the models to specific needs without vendor restrictions\[2\].

- **Community Support**: Open-source LLMs often benefit from active community support, fostering innovation and collaboration\[3\].

These models and platforms provide a range of options for developers and organizations looking to leverage LLMs for various applications, from chatbots to code generation. However, it is crucial to consider the ethical implications and potential risks associated with using uncensored models.

Sources
\[1\] Studying Underground Market for Large Language Models ... <https://www.techpolicy.press/studying-black-market-for-large-language-models-researchers-find-openai-models-power-malicious-services/>
\[2\] 8 Top Open-Source LLMs for 2024 and Their Uses - DataCamp <https://www.datacamp.com/blog/top-open-source-llms>
\[3\] Best Open Source LLMs of 2024 - [Klu.ai](http://Klu.ai) <https://klu.ai/blog/open-source-llm-models>
\[4\] A list of open LLMs available for commercial use. - GitHub <https://github.com/eugeneyan/open-llms/activity>
\[5\] You're Missing Out on Open-Source LLMs! - CODE Magazine [https://www.codemag.com/Article/2403041/You’re-Missing-Out-on-Open-Source-LLMs](https://www.codemag.com/Article/2403041/You%E2%80%99re-Missing-Out-on-Open-Source-LLMs)!

Using uncensored large language models (LLMs) on Hugging Face involves several steps, whether you are accessing them directly from the platform or integrating them into your applications. Here's a guide on how to use these models effectively:

## **Accessing Models on Hugging Face**

1. **Create an Account**: To use models on Hugging Face, you need to create a free account. This will allow you to access various models, including gated ones that require permission\[2\].

2. **Explore Models**: Browse the Hugging Face Model Hub to find uncensored LLMs. You can search for specific models or explore collections like Umbra-AI's uncensored LLMs\[6\].

3. **Request Access for Gated Models**: Some models may be gated, requiring you to request access. You can do this by logging into your account and following the instructions on the model's page\[2\].

## **Using Models via API**

1. **Generate an API Token**: After logging in, generate an API token from your account settings. This token will be used to authenticate your requests to the Hugging Face API\[1\].

2. **Use the Transformers Library**: Install the Hugging Face Transformers library, which allows you to interact with models programmatically. You can use it to download, fine-tune, and run models directly from the platform\[3\].

3. **Make API Calls**: Use the API token to make requests to the models. You can send input prompts and receive generated outputs. The process involves setting up an HTTP request with your token and the desired model endpoint\[1\].

## **Running Models Locally**

1. **Download Models**: If you prefer to run models locally, you can download them using the Transformers library. This allows you to fine-tune them on your own datasets and integrate them into your applications without relying on external servers\[3\].

2. **Setup Environment**: Ensure your environment is set up with the necessary dependencies, such as Python and the Transformers library. This setup will enable you to load and run models on your local machine.

## **Considerations**

- **Cost**: While many models are free to use, some may incur costs, especially if accessed through high-demand APIs or if they require significant computational resources\[4\].

- **Ethical Use**: Uncensored models may not have built-in safety features, so it's essential to implement your own safeguards to prevent misuse.

By following these steps, you can effectively use uncensored LLMs from Hugging Face for various applications, whether for experimentation, development, or production use.

Sources
\[1\] How to Access HuggingFace Models with API - YouTube <https://www.youtube.com/watch?v=n28awivN2FA>
\[2\] Gated models - Hugging Face <https://huggingface.co/docs/hub/en/models-gated>
\[3\] How to Get Started with Hugging Face – Open Source AI Models ... <https://www.freecodecamp.org/news/get-started-with-hugging-face/>
\[4\] How to Host an Uncensored AI Model in Budget - Stack Overflow <https://stackoverflow.com/questions/77257861/how-to-host-an-uncensored-ai-model-in-budget>
\[5\] Uncensor any LLM with abliteration - Hugging Face <https://huggingface.co/blog/mlabonne/abliteration>
\[6\] LLMs \[UNCENSORED\] - a Umbra-AI Collection - Hugging Face <https://huggingface.co/collections/Umbra-AI/llms-uncensored-6531951e548eae0c99f4a534>

Running uncensored large language models (LLMs) on macOS can be done using several methods and tools. Here’s a guide on how to set up and use these models on your Mac:

## **Setting Up Your Mac for LLMs**

1. **Install Homebrew**: This package manager simplifies the installation of software on macOS. You can install it by running the following command in your terminal:

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and Jupyter Notebook**: These are essential for running scripts and managing your LLM projects.

   ```bash
   brew install python
   pip3 install jupyter
   ```

3. **Install Required Packages**: Depending on the model you want to use, you may need to install additional packages like `transformers` and `torch`.

   ```bash
   pip3 install transformers torch
   ```

## **Running Models Locally**

### **Using Ollama**

Ollama is a tool that allows you to run models like LLaMA 2 locally on your Mac:

- **Download and Install Ollama**: Once downloaded, drag the Ollama application into your Applications folder.

- **Run Models**: Use the terminal to load and run models. For example, to load LLaMA 2, you might use:

   ```bash
   ollama pull llama2
   ollama run llama2
   ```

### **Using Private LLM**

Private LLM is an app that allows you to run models locally without internet access:

- **Download Private LLM**: Available from the App Store, it supports various models and integrates with macOS features like Apple Shortcuts for custom workflows.

- **Run Models**: You can use Private LLM for uncensored interactions, including NSFW content, by selecting models like Dolphin 2.6 Mixtral 8x7b.

### **Using Llama.cpp**

Llama.cpp is a project that provides optimized C++ code for running LLMs:

- **Install Llama.cpp**: Use Homebrew or another package manager to set up the environment.

- **Download and Run Models**: Fetch models from sources like Hugging Face and use commands to run them locally.

## **Considerations**

- **Hardware Requirements**: Running LLMs locally can be resource-intensive. Macs with Apple Silicon CPUs (M1/M2) are particularly well-suited for these tasks due to their efficient architecture.

- **Ethical Use**: Uncensored models may not have content filters, so it's important to use them responsibly, especially when engaging in sensitive topics.

By following these steps, you can effectively run uncensored LLMs on your macOS device, leveraging the power of AI for various applications without relying on cloud services.

Sources
\[1\] A Comprehensive Guide to Run LLMs on Your Macbook <https://www.pacenthink.io/post/setup-llm-model-locally-macbook/>
\[2\] How to run large language models (LLAMA2, MISTRAL, PHI) on ... <https://www.youtube.com/watch?v=awkvqNO55sE>
\[3\] Run Llama 2 on your own Mac using LLM and Homebrew <https://simonwillison.net/2023/Aug/1/llama-2-mac/>
\[4\] Local LLMs Part 1 – Apple MacOS - iuvo Technologies <https://blogs.iuvotech.com/local-llms-part-1-apple-macos>
\[5\] Install MLC LLM Python Package <https://llm.mlc.ai/docs/install/mlc_llm>
\[6\] Run Local LLMs on iPhone or Mac Easily Using Private LLM <https://privatellm.app/blog/run-local-gpt-on-ios-complete-guide>
\[7\] domschl/HuggingFaceGuidedTourForMac: A guided tour ... - GitHub <https://github.com/domschl/HuggingFaceGuidedTourForMac>
\[8\] Best Open Source LLMs of 2024 <https://klu.ai/blog/open-source-llm-models>
\[9\] Uncensored AI Chatbot for iPhone/Mac: No Login or Tracking <https://privatellm.app/blog/best-uncensored-llms-no-login-no-subscription-no-surveillance>

Using large language models (LLMs) on iOS devices, where the models are hosted elsewhere, can be achieved through several applications and services. Here are some options and methods to consider:

## **Using Hosted LLMs on iOS**

### **1\. Private LLM**

- **Overview**: Private LLM is an app that allows users to run LLMs directly on their iPhone or iPad. It emphasizes privacy by operating offline and does not require an API key or internet connection for basic functionalities.

- **Features**: The app supports various models and integrates with iOS features like Apple Shortcuts, allowing users to create custom workflows. It offers a one-time purchase model, enabling users to download new models for free indefinitely\[3\]\[4\].

### **2\. LLM Farm**

- **Overview**: LLM Farm is another app available for iOS and macOS that supports a wide range of LLMs. It allows users to load and configure different models with various parameters\[5\].

- **Features**: The app supports multiple inference methods and sampling techniques, and it is compatible with a variety of models, including LLaMA, GPTNeoX, and more. It also supports LoRA adapters for fine-tuning and exporting models.

## **Considerations for Using LLMs on iOS**

- **Internet Dependency**: While some apps like Private LLM offer offline capabilities, using hosted LLMs typically requires an internet connection to access the models remotely.

- **Privacy**: Apps like Private LLM prioritize user privacy by not collecting data or requiring registration, making them suitable for sensitive applications.

- **Integration**: iOS apps can integrate with system features like Apple Shortcuts to automate tasks and enhance functionality, providing a seamless user experience.

By using these applications, iOS users can leverage the power of LLMs for various tasks, from creative writing to role-playing, while maintaining control over privacy and data security.

Sources
\[1\] LLM as a System Service on Mobile Devices - arXiv <https://arxiv.org/html/2403.11805v1>
\[2\] Understanding the cost of Large Language Models (LLMs) <https://www.tensorops.ai/post/understanding-the-cost-of-large-language-models-llms>
\[3\] Run Local LLMs on iPhone or Mac Easily Using Private LLM <https://privatellm.app/blog/run-local-gpt-on-ios-complete-guide>
\[4\] Uncensored AI Chatbot for iPhone/Mac: No Login or Tracking <https://privatellm.app/blog/best-uncensored-llms-no-login-no-subscription-no-surveillance>
\[5\] LLM Farm on the App Store <https://apps.apple.com/us/app/llm-farm/id6461209867>

To use an uncensored large language model (LLM) hosted on Hugging Face via apps on both a MacBook and an iPhone, you can follow these detailed steps. These steps involve using different apps for each platform, and you are open to paying for both the apps and the inference costs on Hugging Face.

## **For MacBook: Using Ollama**

1. **Install Ollama**: Ollama is a tool that allows you to run Hugging Face models locally on your MacBook. You can download and install it from the Ollama website.

2. **Set Up Hugging Face Account**: Ensure you have a Hugging Face account with a valid payment method for accessing inference endpoints. This will allow you to deploy models and pay for the compute time needed to run them\[1\]\[4\].

3. **Download Models**: Use Ollama to pull and run models from Hugging Face. For example:

   ```bash
   ollama pull llama2
   ollama run llama2
   ```

   This command will download and run the LLaMA 2 model on your MacBook\[2\]\[3\].

4. **Configure Inference Endpoints**: If you prefer using Hugging Face's hosted inference endpoints, configure your API access and use the Hugging Face CLI to interact with the models.

## **For iPhone: Using Private LLM**

1. **Download Private LLM**: Available on the App Store, Private LLM allows you to run models directly on your iPhone. It's a one-time purchase app that provides access to various uncensored models\[5\]\[6\].

2. **Select and Use Models**: Within Private LLM, you can choose from a range of models like Dolphin 2.9 or Kappa-3 Phi Abliterated, which are designed for uncensored interactions\[6\].

3. **Integration with iOS Features**: Leverage Apple Shortcuts to automate tasks and create custom workflows with Private LLM. This allows you to integrate AI functionalities into your daily routines seamlessly\[7\].

4. **Offline Functionality**: While Private LLM offers offline capabilities, you can also connect it to Hugging Face's hosted models if needed, using API endpoints for more complex tasks.

## **General Considerations**

- **Inference Costs**: Be aware of the hourly costs associated with using Hugging Face's inference endpoints, which can vary based on CPU or GPU usage\[4\].

- **Privacy and Security**: Private LLM emphasizes privacy by running models locally without internet access, ensuring no data is collected or tracked\[6\].

- **Model Selection**: Choose models that fit your specific needs, whether for creative tasks, coding assistance, or other applications.

By following these steps, you can effectively use uncensored LLMs on both your MacBook and iPhone, leveraging the capabilities of Hugging Face's platform and specialized apps tailored for each device.

Sources
\[1\] Hugging Face Alternatives & Pricing - Sprout24 <https://sprout24.com/hub/hugging-face/>
\[2\] Hugging Face GGUF Models locally with Ollama - YouTube <https://www.youtube.com/watch?v=7BH4C6-HP14>
\[3\] How to run hugging face models easily with Ollama - Reddit <https://www.reddit.com/r/ArtificialInteligence/comments/1cgvyxl/how_to_run_hugging_face_models_easily_with_ollama/>
\[4\] Access Inference Endpoints - Hugging Face <https://huggingface.co/docs/inference-endpoints/en/guides/access>
\[5\] Private LLM - Local AI Chatbot on the App Store <https://apps.apple.com/us/app/private-llm-local-ai-chatbot/id6448106860>
\[6\] Uncensored AI Chatbot for iPhone/Mac: No Login or Tracking <https://privatellm.app/blog/best-uncensored-llms-no-login-no-subscription-no-surveillance>
\[7\] Run Local LLMs on iPhone or Mac Easily Using Private LLM <https://privatellm.app/blog/run-local-gpt-on-ios-complete-guide>
\[8\] GitHub - domschl/HuggingFaceGuidedTourForMac: A guided tour on how to use HuggingFace large language models on Macs with Apple Silicon <https://github.com/domschl/HuggingFaceGuidedTourForMac>

To use uncensored large language models (LLMs) hosted on Hugging Face via apps on both a MacBook and an iPhone, you can subscribe to a service that provides hosted inference and use compatible apps to connect to these services. Here's a detailed guide on how to achieve this:

## **Hosted Inference Services**

### **Hugging Face Inference API**

- **Overview**: Hugging Face provides a robust API for hosted inference, allowing you to run models without hosting them locally. You can subscribe to their service to access various models, including uncensored ones.

- **Subscription**: You'll need to create a Hugging Face account and subscribe to their paid plans, which offer different levels of access based on your usage needs. This will cover the inference costs associated with running the models.

## **Apps for MacBook and iPhone**

### **MacBook: LLMFarm**

- **LLMFarm**: This app allows you to work with various LLMs on macOS. While it supports local model loading, it can also be configured to connect with hosted models through APIs.

- **Features**: LLMFarm supports multiple inferences and sampling methods, making it versatile for different applications. It integrates with macOS features for enhanced functionality\[5\]\[6\].

### **iPhone: Hugging Chat**

- **Hugging Chat**: This app is available on the App Store and allows you to interact with open-source models hosted on Hugging Face. It provides a user-friendly interface to choose and customize AI models for various tasks\[8\].

- **Integration**: Hugging Chat syncs with the online version, ensuring that your interactions are consistent across devices. It offers privacy features by not storing chats or using them for training new AIs\[8\].

## **Steps to Connect and Use**

1. **Set Up Hugging Face Account**: Ensure you have a Hugging Face account with a valid subscription plan to access their hosted models.

2. **Install Apps**: Download and install LLMFarm on your MacBook and Hugging Chat on your iPhone from their respective app stores.

3. **Configure API Access**: Use the API keys provided by Hugging Face to configure the apps. This will allow you to connect to the hosted models seamlessly.

4. **Select Models**: Within the apps, choose the specific uncensored models you want to use. Both apps provide options to customize and interact with these models based on your requirements.

5. **Use and Integrate**: Leverage the features of each app to integrate AI capabilities into your tasks. For example, use Hugging Chat for on-the-go interactions and LLMFarm for more extensive tasks on your MacBook.

By following these steps, you can effectively use uncensored LLMs hosted on Hugging Face across both your MacBook and iPhone, taking advantage of the apps' capabilities to enhance productivity and creativity.

Sources
\[1\] Exploring Uncensored LLM Model – Dolphin 2.9 on Llama-3-8b <https://askaresh.com/2024/05/02/exploring-uncensored-llm-model-dolphin-2-9-on-llama-3-8b/>
\[2\] Large Language Model Development Services (LLM) - Webkul <https://webkul.com/large-language-model-development-services/>
\[3\] Dive Into Uncensored LLMs: All You Need to Know <https://blogs.novita.ai/dive-into-uncensored-llms-all-you-need-to-know/>
\[4\] Private LLM - Local AI Chatbot <https://apps.apple.com/us/app/private-llm-local-ai-chatbot/id6448106860>
\[5\] guinmoon/LLMFarm: llama and other large language models on iOS ... <https://github.com/guinmoon/LLMFarm>
\[6\] LLM Farm <https://apps.apple.com/us/app/llm-farm/id6461209867>
\[7\] Run Local LLMs on iPhone or Mac Easily Using Private LLM <https://privatellm.app/blog/run-local-gpt-on-ios-complete-guide>
\[8\] Hugging Chat on the App Store <https://apps.apple.com/us/app/hugging-chat/id6476778843>



> 

source: <https://www.perplexity.ai/search/research-uncensored-llms-avail-WITr4yGiRKCOHPIwe6QKqQ>

> research uncensored LLMs available, which are hosted and can be used by the....html

source: <https://www.perplexity.ai/search/research-uncensored-llms-avail-WITr4yGiRKCOHPIwe6QKqQ>


