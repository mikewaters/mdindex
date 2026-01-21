---
tags:
  - landscape
---
# Python UI Landscape

Is a:: Technical Landscape [Document.md](./Document.md)

<https://docs.cloud.ploomber.io/en/latest/apps/panel.html> SOMEBODY DID THIS



## Dimensions of Analysis

Per [Deep Research Dimensions.md](./Deep%20Research%20Dimensions.md)

- Python dimensions, that is one of my collection interests and so I define a dimensions list

- GUI apps

Fitness: For any Activity that has declared it needs a gui , and I don’t know how I’ll do this, the analysis should determine fitness for that need. Maybe then use should select something at ingest time.

Collections: The analysis results need to be surfaceable, and so if there is an existing g Landscape Collection then the results should be included in it. Landscape Collections should be serializeable to a single file somehow, to preserve the way I am doing it now. Maybe I can use [MichaelDown.md](./MichaelDown.md) for this, by adding a hashmark to the line in that Note.

## Landscape



<https://github.com/hoffstadt/DearPyGui>

> Dear PyGui: A fast and powerful Graphical User Interface Toolkit for Python with minimal dependencies

### Streamlit

ML

### Dash

BI

[Dash by Plotly](https://dash.plotly.com) stands out primarily for its tight integration with Plotly's visualization library and more granular control over reactivity through callbacks. It offers a more traditional web development approach while still staying Python-centric. Dash is particularly strong in enterprise deployments and when precise control over component interactions is needed.

### Panel

BI

<https://panel.holoviz.org>

Panel provides a more flexible widget system and seamlessly integrates with multiple plotting libraries (Bokeh, Matplotlib, Plotly). Its unique selling point is the ability to serve visualizations in multiple contexts - notebooks, standalone apps, or servers. Panel excels when you need to create dashboards that work across different environments.

### Gradio

ML

<https://github.com/gradio-app/gradio>

Gradio specializes in creating interfaces for machine learning models. Its strength lies in its simplicity for ML model demonstrations - you can create an interface for a model with just a few lines of code. It's particularly popular for sharing Hugging Face models and quick ML prototypes.

### Shiny

ML, -ish

<https://posit.co/blog/why-shiny-for-python/>

Shiny for Python, inspired by R's Shiny, offers a robust reactive programming model. It's especially appealing to R users transitioning to Python and those needing sophisticated reactive data flows. The framework provides strong statistical computing features.

<https://shiny.posit.co/py/>



### Shiny Express

ML

<https://shiny.posit.co/blog/posts/shiny-express/>

 addition to Shiny for Python since its inception: **Shiny Express**, a new way to write Shiny apps in Python

### NiceGUI

<https://github.com/zauberzeug/nicegui>

Nicegui offers a more modern approach with real-time updates and a comprehensive widget set. It's particularly strong in creating responsive layouts and handling real-time data. The framework uses FastAPI under the hood, providing good performance.

### 

### Taipy

BI

<https://github.com/Avaiga/taipy>

Taipy combines data pipeline management with web interface creation. Its distinguishing feature is the tight integration between data workflows and UI components, making it particularly useful for end-to-end data applications where you need both pipeline orchestration and visualization.

### Bokeh

BI

Bokeh servers offer a lower-level approach to creating interactive visualizations. While more complex to set up than Streamlit, they provide fine-grained control over both the visualization and server aspects. Bokeh's strength lies in handling large datasets and creating custom interactive visualizations.

### Reflex (hosted upsell)

<https://github.com/reflex-dev/reflex>

Reflex uses websockets as backend-frontend transport, and compiles Python Ui code to a Reqct app using radix components. Backend is FastAPI 

<https://reflex.dev/blog/2024-03-21-reflex-architecture/#the-reflex-architecture>

Reflex (formerly Pynecone) aims to create high-performance web apps using Python with a React-like development experience. It's designed for building full-stack applications and offers better performance than Streamlit for complex applications, though with a steeper learning curve.

Has AI Assisted development, seems like part of monetization.

Unsure if this is something for me

### Rio

<https://github.com/rio-labs/rio>

**Rio** is an easy to use framework for creating websites and apps and is based **entirely on Python**.

<https://news.ycombinator.com/item?id=41567711>

### Oobabooga

<https://github.com/oobabooga/text-generation-webui>

### 

## LLM-adjacent

### Mesop (google)

<https://google.github.io/mesop/>

## Node-adjacent

### Nodezator

<https://github.com/IndiePython/nodezator>

### Rete.js +

Rete.js is a JavaScript framework that can be integrated with Python backends. 

<https://github.com/retejs/rete>

Here’s an example of a simple react app that calls into a Python service: <https://github.com/aiidateam/aiida-workgraph-web-ui>

## Jupyter-adjacent

### Mercury

<https://github.com/mljar/mercury>

Mercury provides a different approach by focusing on notebook-based development with added interactivity. It's particularly good for creating interactive reports and dashboards from existing Jupyter notebooks while maintaining their narrative structure.

### Voila

Voilà transforms Jupyter notebooks directly into interactive dashboards. This makes it ideal for data scientists who work primarily in notebooks and want to share their analysis without major modifications. The familiar notebook structure can be both an advantage and limitation.

### Solara

> A Pure Python, React-style Framework for Scaling Your Jupyter and Web Apps
>
> <https://github.com/widgetti/solara/>

<https://solara.dev>

Requires a kernel

## HTML wrappers

### FastHTML

<https://docs.fastht.ml>

### FastUI (pydantic)

<https://github.com/pydantic/FastUI>

<https://fastui-demo.onrender.com>

### Htpy

<https://github.com/pelme/htpy>

Generate HTML in Python

### PyWebUI

<https://github.com/pywebio/PyWebIO>

PyWebIO offers a unique approach focusing on turning Python functions into web interfaces. It's particularly good for creating form-like interfaces and wizard-style applications where you need to guide users through a sequence of steps.

## Cross-platform

### Kivy

<https://github.com/kivy/kivy>

Open source UI framework written in Python, running on Windows, Linux, macOS, Android and iOS



Dashnoarding that’s somehow integrates with an LLM provider for the enterprise

<https://github.com/h2oai/wave>

Aaaand they have an iPhone assistant <https://apps.apple.com/us/app/h2o-ai-personal-gpt/id6504365990>

Nice guy built on fast api

<https://nicegui.io/>

<https://justpy.io/>