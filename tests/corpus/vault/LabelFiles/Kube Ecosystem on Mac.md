---
tags:
  - document 📑
---
# Kube Ecosystem on Mac

Related: [Container Orchestration on MacOS.md](./Container%20Orchestration%20on%20MacOS.md) [Container and VM Orchestration (Proxmox et al).md](./Container%20and%20VM%20Orchestration%20\(Proxmox%20et%20al\).md)

> These all require docker containers, and use a Linux VM for installing the containers.

### K-something

#### k3s

> <https://k3s-io.github.io/>
>
> *Ultra-lightweight Kubernetes distribution* by Rancher Labs.

- Designed for resource-constrained environments, edge, and IoT.

- Single binary (<100MB), minimal dependencies, and quick installation.

- Uses SQLite by default for cluster state, but supports other databases.

- Production-ready, but also great for local dev due to low resource usage

#### k0s

> k0s is the simple, solid & certified Kubernetes distribution that works on any infrastructure: bare-metal, on-premises, edge, IoT, public & private clouds. It's 100% open source & free.\
> <https://k0sproject.io/>

- *Lightweight, zero-friction Kubernetes distribution*.

- Production-ready and CNCF conformant.

- Minimal resource requirements and supports multi-node clusters.

- Easy to install and manage, suitable for both local development and edge/production

#### kind (**Kubernetes IN Docker)**

> [kind](https://sigs.k8s.io/kind) is a tool for running local Kubernetes clusters using Docker container “nodes”.\
> kind was primarily designed for testing Kubernetes itself, but may be used for local development or CI.
>
> <https://kind.sigs.k8s.io/>

- *Runs Kubernetes clusters in Docker containers*.

- Fast startup and teardown, ideal for CI/CD, ephemeral clusters, and local development.

- Requires Docker to be installed.

- Not intended for production, but excellent for testing and development

#### minikube

> Minikube is more for dev environments than prod. So k0s over it anytime. For dev envs, I adopted KinD, I can even run it in CI for tests.\
> [source](https://news.ycombinator.com/item?id=41604262#41605407)

- *Official Kubernetes local cluster tool*.

- Runs a Kubernetes cluster in a VM (or Docker/Podman/HyperKit).

- Feature-rich, supports add-ons, and offers a "vanilla" Kubernetes experience.

- Slightly heavier on resources compared to kind/k3s/k0s.

- Good for learning, testing, and exploring the full Kubernetes feature set

#### k3d

- Runs k3s clusters in Docker containers.

- Very fast, lightweight, and supports multi-node clusters.

- Great for local development, especially if you want k3s features with Docker convenience

#### microK8s

- Lightweight, single-command install, production-grade, and CNCF conformant.

- Runs natively on macOS (via Homebrew) and supports multi-node clusters.

- Offers many built-in add-ons and automatic updates

**Talos Linux**

Minimal, immutable OS for running Kubernetes, suitable for advanced users and edge/production

**Colima**

Provides container runtimes and Kubernetes on macOS, often used as a Docker Desktop alternative

**Lima**

Linux virtual machines for macOS, can run Kubernetes distributions

## cluster management tools

#### k9s

> k9s is a terminal-based UI for managing Kubernetes clusters, not a tool for running clusters themselves. It works on top of any Kubernetes cluster, including those created by minikube, kind, k3s, etc.

<https://k9scli.io/>

#### rancher desktop

## inbox