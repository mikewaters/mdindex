---
tags:
  - document 📑
---
# Container Orchestration on MacOS

keywords:: Docker, Kubernetes, k8s, k3s, Rancher

related:: [Kube Ecosystem on Mac.md](./Kube%20Ecosystem%20on%20Mac.md) [Kubernetes Alternatives.md](./Kubernetes%20Alternatives.md)

Jan 30, 2025

### Uncloud

> A lightweight tool for deploying and managing containerised applications across a network of Docker hosts. Bridging the gap between Docker and Kubernetes
>
> <https://github.com/psviderski/uncloud>

> Uncloud is a lightweight clustering and container orchestration tool that lets you deploy and manage web apps across cloud VMs and bare metal with minimized cluster management overhead. It creates a secure WireGuard mesh network between your Docker hosts and provides automatic service discovery, load balancing, ingress with HTTPS, and simple CLI commands to manage your apps.
>
> Unlike traditional orchestrators, there's no central control plane and quorum to maintain. Each machine maintains a synchronized copy of the cluster state through peer-to-peer communication, keeping cluster operations functional even if some machines go offline.
>
> Uncloud aims to be the solution for developers who want the flexibility of self-hosted infrastructure without the operational complexity of Kubernetes.

![IMG_7786.webp](./Container%20Orchestration%20on%20MacOS-assets/IMG_7786.webp)

## Container Desktop apps

### Docker Desktop ($)

supports local k8s cluster, docker containers, docker swarm (?)

cant use professionally without a $license (CAI has them but I can’t get one for some reason)

### OrbStack ($)

MacOS app

> OrbStack is the fast, light, and easy way to run Docker containers and Linux. Develop at lightspeed with our Docker Desktop alternative.
>
> <https://orbstack.dev/>

### Rancher Desktop

> An open-source application that provides all the essentials to work with containers and Kubernetes on the desktop
>
> <https://rancherdesktop.io/>\
> <https://github.com/rancher-sandbox/rancher-desktop/>
>
> from SUSE

 supports local k3s

### Podman Desktop

> Podman desktop companion
>
> A cross-platform desktop UI made by the podman team
>
> [container-desktop.com/](container-desktop.com/)
>
> <https://github.com/iongion/container-desktop>

HN [discussion](https://news.ycombinator.com/item?id=41604262)

#### GPU support

Allows containers to access the GPU on a Mac, which is an issue for docker if you want containers to perform inference natively: 

<https://podman-desktop.io/docs/podman/gpu>

> Currently, Docker does not support Metal GPUs.
>
>  When running LLMs on Docker with an Apple M3 or M4 chip, they will operate in CPU mode regardless of the chip's class, as Docker only supports Nvidia and Radeon GPUs.
>
> If you're developing LLMs on Docker, consider getting a Framework laptop with an Nvidia or Radeon GPU instead.
>
> Source: I develop an AI agent framework that runs LLMs inside Docker on an M3 Max (<https://kdeps.com>).
>
> source: <https://news.ycombinator.com/item?id=43278163>

### Flox

May be more of a Docker alt? not sure, worth a try though

> Create development environments with all the dependencies you need and easily share them with colleagues. Work consistently across the entire software lifecycle.\
> <https://flox.dev/docs/>\
> <https://github.com/flox/flox>

## TUI apps

### Lazydocker

> A TUI app for managing docker. 
>
> <https://github.com/jesseduffield/lazydocker>

To use with Colima, I need to provide it with an alternate unix domain socket (which I am doing in an \~/.zshrc alias):

`DOCKER_HOST=unix:///Users/snip/.colima/default/docker.sock lazydocker […]`

Install: `brew install jesseduffield/lazydocker/lazydocker`

## Container Runtimes

### Docker Engine

### Colima

> Container runtimes on macOS (and Linux) with minimal setup
>
> <https://github.com/abiosoft/colima>

Creates a VM locally to run containers.

> The default VM created by Colima has 2 CPUs, 2GiB memory and 100GiB storage.

Support for multiple container runtimes

- ++[Docker](https://docker.com/)++ (with optional Kubernetes)

- ++[Containerd](https://containerd.io/)++ (with optional Kubernetes)

- ++[Incus](https://linuxcontainers.org/incus)++ (containers and virtual machines)

#### Notes

Install using brew, but don’t use brew services to keep it running.

Requires a `—network-address` param, or a similar config to be made in order for it to listen on an IP. `colima start —network-address` or `colima start —edit` and then make the change.

#### Technical

> Colima means Containers on ++[Lima](https://github.com/lima-vm/lima)++.

> Lima is aka Linux Machines

> LIMA uses MacOS native virtualization to provide linux environments, it is like WSL, but for MacOS.
>
> Lima launches Linux virtual machines with automatic file sharing and port forwarding
>
> <https://lima-vm.io/>

# **Notes**

Articles:

- <https://www.swyx.io/running-docker-without-docker-desktop>

- <https://www.cncf.io/blog/2023/02/02/docker-on-macos-is-slow-and-how-to-fix-it/>

<https://github.com/dstackai/dstack>

Discussion: https://news.ycombinator.com/item?id=43714959

<https://rkiselenko.dev/blog/development-on-mac-with-utm/development-on-mac-with-lima/>

\## Lime ecosystem

<https://github.com/trycua/cua/tree/main/libs/lumier>

Breakdown of container-> hardware relationships - Lume, Lima etc

<https://github.com/trycua/cua/issues/10>

Birutalizing MacOS on MacOS

<https://github.com/insidegui/VirtualBuddy>

<https://github.com/utmapp/UTM>

<https://github.com/trycua/cua/issues/10>

<https://github.com/lima-vm/lima>

<https://github.com/cirruslabs/tart>

<https://github.com/trycua/cua/tree/main/libs/lumier>