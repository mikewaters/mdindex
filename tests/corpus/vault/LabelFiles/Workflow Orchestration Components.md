---
tags:
  - research
  - landscape
Document Type:
  - Technical
Topic:
  - software
Subject:
  - Solution
---
# Workflow Orchestration Components

Software frameworks, components, and libraries for running pipelines

Related: [Frontend Whiteboard, Node(-y) Tools.md](./Frontend%20Whiteboard,%20Node\(-y\)%20Tools.md), [Visual Programming Tools.md](./Visual%20Programming%20Tools.md)

## Solutions

Can break this down into a few layers or types: Task execution, Workflow assembling, and then actual execution engines (like a bunch of servers).



### Flyte (AI-focused)

> Build & deploy data & ML pipelines, hassle-free
>
> The infinitely scalable and flexible workflow orchestration platform that seamlessly unifies data, ML and analytics stacks.

> Flyte is an open-source orchestrator that facilitates building production-grade data and ML pipelines. It is built for scalability and reproducibility, leveraging Kubernetes as its underlying platform. With Flyte, user teams can construct pipelines using the Python SDK, and seamlessly deploy them on both cloud and on-premises environments, enabling distributed processing and efficient resource utilization.

> <https://github.com/flyteorg/flyte>

> <https://flyte.org/>

k8s, python sdk

### Temporal

> Temporal is a durable execution platform that enables developers to build scalable applications without sacrificing productivity or reliability. The Temporal server executes units of application logic called Workflows in a resilient manner that automatically handles intermittent failures, and retries failed operations.

> <https://github.com/temporalio/temporal>

> <https://temporal.io>

Fork of Cadence from Uber

Focuses more on durable execution/resiliency over anything else, overkill for me

### Prefect

> Pythonic orchestration for modern teams.
>
> Meet the workflow engine replacing Airflow at Cash App and Progressive.

> Prefect is an open-source orchestration engine that turns your Python functions into production-grade data pipelines with minimal friction. You can build and schedule workflows in pure Python—no DSLs or complex config files—and run them anywhere. Prefect handles the heavy lifting for you out of the box: automatic state tracking, failure handling, real-time monitoring, and more.
>
> <https://docs.prefect.io/v3/get-started/index>

> <https://github.com/PrefectHQ/Prefect>

Python, has a UI tool

### GCP CDloud Composer

> GCP Dataflow is based on Apache Beam. Cloud Composer is the GCP's Airflow distribution.

> GCP Cloud Composer now integrates with Vertex AI PaaS

+ ### Windmill

   <https://www.windmill.dev/docs/intro>

   > Windmill is a fast, **[open-source](https://github.com/windmill-labs/windmill)** workflow engine and developer platform. It's an alternative to the likes of Retool, Superblocks, n8n, Airflow, Prefect, and Temporal, designed to **build comprehensive internal tools** (endpoints, workflows, UIs). 

   > An [execution runtime](https://www.windmill.dev/docs/script_editor) for scalable, low-latency function execution across a worker fleet.
   >
   > - An [orchestrator](https://www.windmill.dev/docs/flows/flow_editor) for assembling these functions into efficient, low-latency flows, using either a low-code builder or YAML.
   >
   > - An [app builder](https://www.windmill.dev/docs/apps/app_editor) for creating data-centric dashboards, utilizing low-code or JS frameworks like React.

   > ![image 4.png](./Workflow%20Orchestration%20Components-assets/image%204.png)
   >
   > 

+ ### Apache Airflow (\~+ Domino)

   > Airflow™ is a platform created by the community to programmatically author, schedule and monitor workflows.

   Domino frontend, run LLM tasks using Airflow: <https://github.com/Tauffer-Consulting/domino>

   <https://docs.domino-workflows.io/development/domino-py>

   

### DAGSter

I really like these

> I would look at Prefect, Flyte, Dagster, and Temporal ahead of Airflow, though
>
> running Dagster for \~6 months on ECS and it has been rock solid. 
>
> source: <https://news.ycombinator.com/item?id=41227829>

#### Hamilton

> Hamilton is a general-purpose framework to write dataflows using regular Python functions. At the core, each function defines a transformation and its parameters indicates its dependencies. Hamilton automatically connects individual functions into a [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph) (DAG) that can be executed, visualized, optimized, and reported on. Hamilton also comes with a [UI](https://hamilton.dagworks.io/en/latest/hamilton-ui/) to visualize, catalog, and monitor your dataflows.

Task and workflow definitions for python. Lots of LLM integrations. Has a [DAG and code viewer UI](https://hamilton.dagworks.io/en/latest/hamilton-ui/ui/).

#### Flower Power

<https://github.com/legout/flowerpower>

> FlowerPower is a simple workflow framework based on two fantastic Python libraries:
>
> - **++[Hamilton](https://github.com/DAGWorks-Inc/hamilton)++**: Creates DAGs from your pipeline functions
>
> - **++[APScheduler](https://github.com/agronholm/apscheduler)++**: Handles pipeline scheduling

#### Burr

Executes higher level orchestration for Hamilton.



Dash is a Python based task executor, supporting dataframes and arbitrary structures like graphs or arrays. It runs locally or can be deployed to wherever. Simple version of spark. Has a dashboard UI and lots of examples of embarrassing parallelism lol

<https://examples.dask.org/>