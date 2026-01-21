# Container Orchestration Tools GPU Support on macOS

## Tools That Support Direct GPU Access on macOS

### **Podman with libkrun** ✅

Podman is currently the **only major container orchestration tool** that supports direct GPU access on macOS\[1\]\[2\]\[3\]. This is achieved through:

- **libkrun hypervisor**: Replaced Apple's Virtualization.framework as the backend

- **Virtio-GPU Venus**: Provides virtualized GPU acceleration using Vulkan compute shaders through MoltenVK and Metal API

- **Version requirements**: Podman Desktop 1.12+ and Podman 5.2.0+

- **Setup**: Requires creating a new Podman machine with libkrun provider enabled

The GPU support works by creating a virtualized GPU (Virtio-GPU Venus) that translates Vulkan calls to Apple's Metal API, allowing containers to access the host's Apple Silicon GPU for AI workloads\[4\]\[5\].

### **HashiCorp Nomad** ✅

Nomad provides built-in GPU support across platforms, including macOS\[6\]\[7\]\[8\]:

- **Native GPU detection**: Uses device plugins to automatically detect and utilize GPU resources

- **Cross-platform support**: Works on Linux, Windows, and macOS

- **Device plugin system**: Built-in NVIDIA GPU plugin (though limited to NVIDIA on most platforms)

- **Multi-workload support**: Supports containerized and non-containerized applications

However, on macOS specifically, GPU support would be limited by the underlying containerization layer (likely requiring Docker or Podman with proper GPU passthrough).

## Tools That Do **NOT** Support Direct GPU Access on macOS

### **Docker Desktop** ❌

Docker Desktop explicitly does not support GPU access on macOS\[9\]\[10\]\[11\]\[12\]:

- **Apple Virtualization.framework limitation**: Apple hasn't provided an open GPU API for their mandatory virtualization engine

- **No GPU passthrough**: Cannot access the host GPU from within Docker containers

- **Performance impact**: GPU-intensive workloads fall back to CPU-only operation

- **Future potential**: Docker may adopt libkrun in the future, but no timeline announced

### **Kubernetes Distributions** ❌

Most Kubernetes distributions on macOS lack GPU support:

**minikube**: Explicitly states that macOS drivers don't support GPU passthrough\[13\]:

- **Driver limitations**: Supported macOS drivers (hyperkit, qemu, etc.) don't support GPU passthrough

- **Linux-only**: GPU tutorials are Linux-specific with NVIDIA focus

**K3s**: While K3s supports GPU on Linux, it doesn't work natively on macOS\[14\]\[15\]:

- **VM requirement**: Must run inside Linux VMs on macOS (via Multipass or similar)

- **No native GPU access**: The VM layer prevents direct GPU access

**MicroK8s**: Similar limitations to other Kubernetes distributions on macOS\[16\].

### **OrbStack** ❌

OrbStack currently does not support GPU acceleration\[4\]\[17\]:

- **Feature request**: Active GitHub issue requesting libkrun GPU acceleration support

- **No timeline**: No official timeline for GPU support implementation

- **Alternative suggested**: Users directed to Podman for GPU workloads

### **Lima** ❌

Lima doesn't provide GPU acceleration capabilities\[18\]\[19\]:

- **Basic virtualization**: Focuses on simple Linux VM creation and management

- **No GPU passthrough**: No mention of GPU acceleration features in documentation

- **Limited scope**: Designed for basic development environments, not GPU-intensive workloads

### **Docker Swarm** ❌

Docker Swarm has fundamental limitations with GPU support even on Linux\[20\]\[21\]\[22\]:

- **No device support**: Swarm mode doesn't support the `devices` key in compose files

- **Generic resources workaround**: Complex workarounds required using generic resources

- **macOS limitations**: Same underlying Docker limitations apply on macOS

### **Apache Mesos** ❌

While Mesos supports GPU on Linux\[23\]\[24\], it has limited macOS support:

- **Linux-focused**: GPU documentation is Linux and NVIDIA-specific

- **macOS limitations**: No specific GPU support mentioned for macOS deployments

### **Rancher Desktop** ❌

Rancher Desktop doesn't currently support GPU acceleration on macOS\[25\]\[26\]:

- **Feature request**: Open GitHub issue requesting GPU support documentation

- **VM limitations**: Uses VMs that don't expose GPU to containers

- **Ollama integration**: Recent versions include Ollama but run it on the host (not in containers) to access GPU

### **Portainer** ❌

Portainer is a management UI that depends on the underlying container runtime:

- **Runtime dependent**: GPU support depends on Docker/Podman capabilities

- **macOS limitations**: Same restrictions as underlying Docker on macOS

## Technical Limitations on macOS

The fundamental challenge for GPU access in containers on macOS stems from several factors\[27\]\[9\]\[10\]:

1. **Apple Virtualization Framework**: Apple's mandatory virtualization layer doesn't provide open GPU APIs

2. **Container Architecture**: Linux containers require VMs on macOS, adding an isolation layer

3. **Metal vs CUDA**: Apple's Metal API differs from industry-standard CUDA, requiring translation layers

4. **Security Model**: macOS security model restricts direct hardware access from virtualized environments

## Recommendations

For GPU-accelerated container workloads on macOS:

1. **Use Podman with libkrun**: Currently the only viable solution for containerized GPU workloads

2. **Native installation**: Run AI/ML applications directly on macOS to access full GPU performance

3. **Cloud alternatives**: Use cloud platforms like Google Colab or Lightning AI for GPU-intensive workloads

4. **Remote development**: Use cloud-based Kubernetes clusters with GPU nodes for development and testing

The landscape may change as Apple potentially adds GPU passthrough support to their Virtualization framework or as more tools adopt libkrun's approach to GPU virtualization.

Sources
\[1\] Podman Desktop 1.12 Release <https://podman-desktop.io/blog/podman-desktop-release-1.12>
\[2\] GPU container access - Podman Desktop <https://podman-desktop.io/docs/podman/gpu>
\[3\] Introducing GPU support for Podman AI Lab - Red Hat Developer <https://developers.redhat.com/articles/2024/09/10/gpu-support-podman-ai-lab>
\[4\] GPU acceleration in containers · Issue #1818 · orbstack ... - GitHub <https://github.com/orbstack/orbstack/issues/1818>
\[5\] Detecting macOS GPUs when using Podman with GPU passthrough <https://github.com/ollama/ollama/issues/5954>
\[6\] Is HashiCorp Nomad Right for Your Container Orchestration? <https://www.devoteam.com/expert-view/is-hashicorp-nomad-a-smart-choice-for-your-container-orchestration/>
\[7\] Running GPU-Accelerated Applications on Nomad - HashiCorp <https://www.hashicorp.com/resources/running-gpu-accelerated-applications-on-nomad>
\[8\] hashicorp/nomad - GitHub <https://github.com/hashicorp/nomad>
\[9\] Why Docker Can't Use macOS GPUs—And What to Do Instead <https://www.youtube.com/watch?v=t9SM1rRZcMY>
\[10\] Does Nested Virtualization on macOS give docker room to use GPU ... <https://www.reddit.com/r/docker/comments/1iy2r02/does_nested_virtualization_on_macos_give_docker/>
\[11\] Apple Silicon GPUs, Docker and Ollama: Pick two. - Chariot Solutions <https://chariotsolutions.com/blog/post/apple-silicon-gpus-docker-and-ollama-pick-two/>
\[12\] Using GPU With Docker: A How-to Guide - DevZero <https://www.devzero.io/blog/docker-gpu>
\[13\] Using NVIDIA GPUs with minikube - Kubernetes <https://minikube.sigs.k8s.io/docs/tutorials/nvidia/>
\[14\] Install K3s on Mac (M1/M2/M3/M4) | Easy K3s Setup for macOS <https://www.youtube.com/watch?v=NZTQ8zdN6PY>
\[15\] Has anyone used k3s for a GPU cluster? : r/kubernetes - Reddit <https://www.reddit.com/r/kubernetes/comments/1cdqdsk/has_anyone_used_k3s_for_a_gpu_cluster/>
\[16\] Add-on: gpu - MicroK8s <https://microk8s.io/docs/addon-gpu>
\[17\] OrbStack · Fast, light, simple Docker & Linux <https://orbstack.dev>
\[18\] VM drivers for supporting macOS guests ( vz , host-user ) #3618 <https://github.com/lima-vm/lima/issues/3618>
\[19\] Lima: a nice way to run Linux VMs on Mac - Julia Evans <https://jvns.ca/blog/2023/07/10/lima--a-nice-way-to-run-linux-vms-on-mac/>
\[20\] Instructions for Docker swarm with GPUs - GitHub Gist <https://gist.github.com/tomlankhorst/33da3c4b9edbde5c83fc1244f010815c>
\[21\] Using NVIDIA GPU with docker swarm started by docker-compose file <https://www.reddit.com/r/docker/comments/mh36w1/using_nvidia_gpu_with_docker_swarm_started_by/>
\[22\] Docker Swarm does not support use of GPU devices - Stack Overflow <https://stackoverflow.com/questions/68051366/docker-swarm-does-not-support-use-of-gpu-devices>
\[23\] Nvidia GPU Support - Apache Mesos <https://mesos.apache.org/documentation/latest/gpu-support/>
\[24\] Apache Mesos and GPUs - page for sungsoo blog <https://sungsoo.github.io/2018/03/02/apache-mesos-and-gpus.html>
\[25\] Support for GPU's on Rancher-Desktop (MAC) #5561 - GitHub <https://github.com/rancher-sandbox/rancher-desktop/issues/5561>
\[26\] Releases · rancher-sandbox/rancher-desktop - GitHub <https://github.com/rancher-sandbox/rancher-desktop/releases>
\[27\] Enabling containers to access the GPU on macOS - Sergio López <https://sinrega.org/2024-03-06-enabling-containers-gpu-macos/>
\[28\] Local Kubernetes development with GPU on Apple Silicon M1? <https://www.reddit.com/r/kubernetes/comments/u9szih/local_kubernetes_development_with_gpu_on_apple/>
\[29\] Best option to run GPU workloads in a container or VM on Apple ... <https://www.reddit.com/r/MacOS/comments/1hjhdip/best_option_to_run_gpu_workloads_in_a_container/>
\[30\] Kubernetes-GPU-Guide/README.md at master - GitHub <https://github.com/Langhalsdino/Kubernetes-GPU-Guide/blob/master/README.md>
\[31\] Schedule GPUs | Kubernetes <https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/>
\[32\] Containers for Apple Silicon Macs work with GPU-accelerated Vulkan <https://github.com/ggml-org/llama.cpp/discussions/8042>
\[33\] Install and Set Up kubectl on macOS - Kubernetes <https://kubernetes.io/docs/tasks/tools/install-kubectl-macos/>
\[34\] Apple Containerization: lightweight Linux VMs for macOs <https://www.theregister.com/2025/06/10/apple_tries_to_contain_itself/>
\[35\] dstack: AI container orchestration for AI teams <https://dstack.ai>
\[36\] Configure NVIDIA GPU monitoring environment in K8S cluster <https://docs.kakaocloud.com/en/tutorial/observability/nvidia-gpu-monitoring>
\[37\] Enable GPU support - Docker Docs <https://docs.docker.com/compose/how-tos/gpu-support/>
\[38\] Docker Desktop: The #1 Containerization Tool for Developers <https://www.docker.com/products/docker-desktop/>
\[39\] Chapter 7 Docker | The Knowledge <https://gdsl-ul.github.io/the_knowledge/docker.html>
\[40\] Quick start · OrbStack Docs <https://orbstack.dev/docs/quick-start>
\[41\] Athena on Lima - ATLAS Software Documentation <https://atlas-software.docs.cern.ch/athena/containers/lima/>
\[42\] Linux machines - OrbStack Docs <https://docs.orbstack.dev/machines/>
\[43\] Installation | Lima <https://lima-vm.io/docs/installation/>
\[44\] lima-vm experience on MacOS : r/Cloud - Reddit <https://www.reddit.com/r/Cloud/comments/qczhdm/limavm_experience_on_macos/>
\[45\] GPU acceleration / libkrun · orbstack · Discussion #1408 - GitHub <https://github.com/orgs/orbstack/discussions/1408>
\[46\] Customizing Lima instance - Podman Desktop <https://podman-desktop.io/docs/lima/customizing>
\[47\] How to utilise M-series MacOS GPU cores in local Podman Kind ... <https://github.com/containers/podman/discussions/25999>
\[48\] Frequently asked questions · OrbStack Docs <https://orbstack.dev/docs/faq>
\[49\] Installing Hashicorp Nomad Client on a Mac - Persistent <https://blog.graywind.org/posts/hashicorp-nomad-client-macos/>
\[50\] Using GPUs - D2iQ Docs <https://archive-docs-old.d2iq.com/mesosphere/dcos/2.0/deploying-services/gpu>
\[51\] How to use Nomad with Nvidia Docker? - gpu - Stack Overflow <https://stackoverflow.com/questions/44004898/how-to-use-nomad-with-nvidia-docker>
\[52\] \[PDF\] Nvidia GPU Support on Mesos: Bridging Mesos Containerizer and ... <https://events.static.linuxfound.org/sites/events/files/slides/mesoscon_asia_gpu_v1.pdf>
\[53\] Docker Swarm GPU Support - General <https://forums.docker.com/t/docker-swarm-gpu-support/148295>
\[54\] HOW-TO Build Mesos on Mac OSX – Eclipse - Code Trips & Tips <https://codetrips.com/2015/04/19/how-to-build-mesos-on-mac-osx-eclipse/comment-page-1/>
\[55\] Getting started with Swarm mode - Docker Docs <https://docs.docker.com/engine/swarm/swarm-tutorial/>
\[56\] Nomad device driver for Nvidia GPU - GitHub <https://github.com/hashicorp/nomad-device-nvidia>
\[57\] Does Apache Mesos recognize GPU cores? - Stack Overflow <https://stackoverflow.com/questions/27872558/does-apache-mesos-recognize-gpu-cores>
\[58\] Docker Swarm on your MAC in 5 minutes | by Park Sehun | Medium <https://sehun.me/docker-swarm-on-your-mac-in-5-minutes-de09c5a88b2f>
\[59\] What is Nomad? | Nomad - HashiCorp Developer <https://developer.hashicorp.com/nomad/docs/what-is-nomad>
\[60\] Apache Mesos <https://mesos.apache.org>
\[61\] Enabling GPU Access in Portainer & Docker - YouTube <https://www.youtube.com/watch?v=Sypi9dfMLX0>
\[62\] A guide on using NVidia GPUs for transcoding or AI in Kubernetes <https://github.com/UntouchedWagons/K3S-NVidia>
\[63\] Help with GPU passthrough for Docker/Portainer - Reddit <https://www.reddit.com/r/docker/comments/17ghl5e/help_with_gpu_passthrough_for_dockerportainer/>
\[64\] Rancher Desktop by SUSE <https://rancherdesktop.io>
\[65\] Stack definition using web editor with Nvidia/GPU support fails to ... <https://github.com/portainer/portainer/issues/5902>
\[66\] Using gpu in fraction in k3s - kubernetes - Stack Overflow <https://stackoverflow.com/questions/77506551/using-gpu-in-fraction-in-k3s>
\[67\] using GPU's with rancher - Reddit <https://www.reddit.com/r/rancher/comments/14ec9s0/using_gpus_with_rancher/>
\[68\] Set up a macOS build environment - Portainer Documentation <https://docs.portainer.io/contribute/build/mac>
\[69\] Adding A GPU node to a K3S Cluster <https://www.radicalgeek.co.uk/adding-a-gpu-node-to-a-k3s-cluster/>
\[70\] Rancher desktop is also a viable option and free. Many including my ... <https://news.ycombinator.com/item?id=42225135>
\[71\] How to Use Your GPU in a Docker Container - Roboflow Blog <https://blog.roboflow.com/use-the-gpu-in-docker/>
\[72\] Working with LLMs using Open WebUI - Rancher Desktop Docs <https://docs.rancherdesktop.io/tutorials/working-with-llms>
\[73\] Install Portainer Agent on Docker Standalone <https://docs.portainer.io/admin/environments/add/docker/agent>
\[74\] Installing k3s on Mac M1? · k3s-io k3s · Discussion #5436 - GitHub <https://github.com/k3s-io/k3s/discussions/5436>
\[75\] Docker on MacOS is still slow? - Paolo Mainardi <https://www.paolomainardi.com/posts/docker-performance-macos-2025/>
\[76\] Setup - Portainer Documentation <https://docs.portainer.io/user/docker/host/setup>
\[77\] cri-o/cri-o: Open Container Initiative-based implementation ... - GitHub <https://github.com/cri-o/cri-o>
\[78\] BigAnteater/MacOS-KVM-GPU-Passthrough - GitHub <https://github.com/BigAnteater/MacOS-KVM-GPU-Passthrough>
\[79\] Chapter 1. Using the CRI-O Container Engine <https://docs.redhat.com/en/documentation/openshift_container_platform/3.11/html/cri-o_runtime/use-crio-engine>
\[80\] GPU Accelerated Containers on Apple Silicon with libkrun and ... <https://www.youtube.com/watch?v=OyTJ8FtQaJ0>
\[81\] How we improved AI inference on macOS Podman containers <https://developers.redhat.com/articles/2025/06/05/how-we-improved-ai-inference-macos-podman-containers>
\[82\] cri-o <https://cri-o.io>
\[83\] Container Runtime Interface (CRI) - Kubernetes <https://kubernetes.io/docs/concepts/architecture/cri/>
\[84\] How to Run PyTorch on a MacOS GPU with Metal <https://apxml.com/posts/pytorch-macos-metal-gpu>
\[85\] Container Runtimes - Kubernetes <https://kubernetes.io/docs/setup/production-environment/container-runtimes/>
\[86\] You can run CUDA, on a Mac ARM GPU, in the browser. | Laurie Kirk <https://www.linkedin.com/posts/laurie-kirk_you-can-run-cuda-on-a-mac-arm-gpu-in-the-activity-7336108121502687232-yTXY>
\[87\] Installing the NVIDIA GPU Operator <https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html>
\[88\] containerd/nerdctl: contaiNERD CTL - Docker-compatible CLI for ... <https://github.com/containerd/nerdctl>
\[89\] Docker vs containerd vs CRI-O: An In-Depth Comparison <https://phoenixnap.com/kb/docker-vs-containerd-vs-cri-o>
\[90\] Installing the NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>

# GPU Capabilities Available to Podman + libkrun on macOS

## 1\. What “GPU access” really means here

Podman’s libkrun provider does **not** pass the physical Apple-Silicon GPU directly into the container. Instead it exposes a **virtual GPU (Virtio-GPU + Venus)** inside the Podman VM. The data path is:

```
Container → Vulkan (user code) ──► Venus front-end
         (or frameworks that speak Vulkan)
Venus messages ──► virglrenderer in the VM
virglrenderer ──► MoltenVK → Apple Metal Shading Language (MSL)
MSL kernels ──► Apple Metal driver → Physical GPU
```

Because the final hop is Metal, any higher-level API that can emit **Vulkan compute or graphics commands** can run, provided it stays within the Vulkan features that MoltenVK and Metal support\[1\]\[2\].

## 2\. Primitives you actually get

| Layer | Exposed to containers? | Notes & limits | 
|---|---|---|
| **Vulkan 1.3 core** | Yes | The guest sees a “virtio-gpu/venus” device offering core Vulkan 1.3 features. | 
| **Vulkan extensions** | Partial | Most compute-oriented extensions that MoltenVK implements (e.g., `VK_KHR_shader_float16_int8`, `VK_KHR_8bit_storage`) are available; extensions requiring tessellation, geometry, or mesh shaders are missing because MoltenVK/Metal lack them\[1\]. | 
| **Metal API** | No (host-side only) | Containers never see `MTLDevice`; Metal is hidden inside MoltenVK. Native Metal apps must run on the macOS host, not in the container. | 
| **CUDA / NVIDIA-specific** | No | There is no CUDA on Apple GPUs, and NVIDIA pass-through is impossible on macOS VMs\[2\]. | 
| **OpenCL / MLX (Metal Performance Shaders for ML)** | Indirect | ML frameworks such as MPSGraph that sit on top of Metal are not exposed; you must use frameworks that target Vulkan (e.g., `ggml-vulkan`, `llama.cpp` Vulkan backend)\[1\]. | 
| **Direct `libGL` / OpenGL** | Experimental | Mesa’s virtio-gpu OpenGL path exists but Apple’s OpenGL drivers are deprecated; performance is lower than Vulkan. | 

## 3\. Practical consequences for developers

### Works well

- Vulkan compute kernels (e.g., `spirv` generated by **TensorFlow-Lite Vulkan delegate**, **llama.cpp ggml-vulkan**, **vkfft**, etc.)  

- Graphics workloads that fit MoltenVK’s feature set (most 2-D/3-D engines that avoid geometry shaders).

### Does **not** work

- Code that needs CUDA, cuDNN, ROCm, or NVIDIA device files (`/dev/nvidia*`).  

- Apps written directly against Metal, Metal Performance Shaders (MPS), or MLX: those frameworks are host-only.  

- Vulkan extensions unimplemented by MoltenVK (e.g., mesh shaders, cooperative matrix) show as unsupported and calls will fail\[1\].

## 4\. Resource and isolation limits

1. **Memory** – The virtual GPU uses system memory inside the VM for command buffers and shared textures. libkrun currently caps a guest to **≤16 GB** RAM; allocate enough via `podman machine init --memory`.  

2. **vCPUs** – libkrun’s TDX flavor is limited to 1 vCPU and ≤3 GiB RAM\[3\]; use the standard flavor for multi-core workloads.  

3. **Device node** – Containers must be started with `--device /dev/dri` (or via CDI spec) so the guest sees the Virtio-GPU device\[4\].  

4. **Performance overhead** – Every API call traverses Venus → virglrenderer → MoltenVK layers; real-world inference runs at \~75% of native Metal speed\[1\].  

5. **Security sandbox** – The GPU is namespaced per-VM. Other macOS processes (including other Podman VMs) cannot peek into your GPU memory.

## 5\. Summary

- Inside a libkrun Podman container you get a **Vulkan 1.3 virtual GPU**; everything else (Metal, MLX, CUDA) is either host-only or unsupported.  

- Run workloads that already have a Vulkan backend; expect \~75% of native Metal performance with slight feature gaps.  

- Allocate enough guest RAM and add `/dev/dri` to containers; otherwise GPU devices will be missing.

With these constraints, Podman + libkrun is a solid option for Vulkan-based AI inference and general compute on Apple-Silicon Macs, but not a drop-in replacement for CUDA or native Metal development.

Sources
\[1\] How we improved AI inference on macOS Podman containers <https://developers.redhat.com/articles/2025/06/05/how-we-improved-ai-inference-macos-podman-containers>
\[2\] GPU container access - Podman Desktop <https://podman-desktop.io/docs/podman/gpu>
\[3\] containers/libkrun: A dynamic library providing ... - GitHub <https://github.com/containers/libkrun>
\[4\] How to utilise M-series MacOS GPU cores in local Podman Kind ... <https://github.com/containers/podman/discussions/25999>
\[5\] Rust libraries | Everything I Know - Wiki <https://wiki.nikiv.dev/programming-languages/rust/rust-libraries/>
\[6\] GPU acceleration in containers · Issue #1818 · orbstack ... - GitHub <https://github.com/orbstack/orbstack/issues/1818>
\[7\] Introducing GPU support for Podman AI Lab - Red Hat Developer <https://developers.redhat.com/articles/2024/09/10/gpu-support-podman-ai-lab>
\[8\] GPU Accelerated Containers on Apple Silicon with libkrun and ... <https://www.youtube.com/watch?v=OyTJ8FtQaJ0>
\[9\] Currently, Docker does not support Metal GPUs. When running ... <https://news.ycombinator.com/item?id=43278163>
\[10\] Podman 5.2 Enhances macOS VMs With GPU Support - Linux Today <https://www.linuxtoday.com/news/podman-5-2-enhances-macos-vms-with-gpu-support/>
\[11\] Enabling containers to access the GPU on macOS - Sergio López <https://sinrega.org/2024-03-06-enabling-containers-gpu-macos/>
\[12\] Podman Desktop blog! | Podman Desktop <https://podman-desktop.io/blog>

# Running llama.cpp in a Podman + libkrun container (Apple-Silicon macOS)

The goal is a container that (a) boots under Podman’s libkrun provider and (b) sees the virtual Vulkan GPU device so llama.cpp can use its Vulkan backend.

## 1\. Prerequisites on the host

| Requirement | Command / Action | 
|---|---|
| Podman Desktop 1.12 + or Podman CLI 5.2 + | Install from [podman.io](http://podman.io) | 
| libkrun provider enabled | `podman machine init --provider=krun --memory 12 --cpus 6 --rootful --now` | 
| Vulkan SDK headers (for host-side build caching) | `brew install vulkan-headers` | 
| Model file (e.g., `llama-7b-v2.Q4_K_M.gguf`) stored on host | Place under `~/models` | 

> libkrun machines are isolated VMs; their RAM/CPU values are fixed at `podman machine init` time. Give **≥10 GiB RAM** for 7 b models and **≥4 vCPUs** for decent throughput.

## 2\. Build a llama.cpp image with Vulkan support

Use a multi-stage Dockerfile so the final image is tiny:

```dockerfile
# Stage 1 – build
FROM docker.io/library/alpine:3.20 AS build
RUN apk add --no-cache build-base cmake git vulkan-headers \
    && git clone --depth 1 https://github.com/ggerganov/llama.cpp /src
WORKDIR /src
# Enable Vulkan backend; turn off CUDA/Metal so the Makefile picks Vulkan
ENV LLAMA_CUBLAS=OFF LLAMA_METAL=OFF LLAMA_VULKAN=ON \
    LLAMA_CLBLAST=OFF LLAMA_BLAS=ON
RUN make -j$(nproc) LLAMA_BUILD=release \
    && mkdir /out && cp -a main/* libllama.so ./build/bin/* /out/

# Stage 2 – runtime
FROM docker.io/library/alpine:3.20
RUN apk add --no-cache libstdc++ libgcc vulkan-loader
COPY --from=build /out /llama
WORKDIR /llama
ENTRYPOINT ["./main"]
```

Build the image (on macOS host):

```bash
podman build -t llama-vulkan .
```

## 3\. Start the container with GPU device exposed

The virtual GPU appears as a DRM node in the **guest** VM. Expose it to the container with `--device`:

```bash
podman run --rm \
  --device /dev/dri \
  -v ~/models:/models:ro \
  llama-vulkan \
  -m /models/llama-7b-v2.Q4_K_M.gguf \
  -ngl 32 -n 128
```

Explanation of key flags  

- `--device /dev/dri` – passes the virtio-gpu DRM device into the container.  

- `-v ~/models:/models:ro` – mounts model files read-only.  

- `-ngl 32` – number of GPU layers; tune for VRAM (virtio-gpu shares host RAM).  

- `-n 128` – generate 128 tokens.

If everything is wired correctly you’ll see a line near startup:

```
backend = vulkan, device = virtio-gpu Venus …
```

## 4\. Troubleshooting

| Symptom | Fix | 
|---|---|
| `No Vulkan devices found` | Ensure the VM was created with **krun** provider and you passed `/dev/dri`. | 
| Segfault when compiling shaders | Use latest Podman and llama.cpp `master`; older MoltenVK lacked some SPIR-V ops. | 
| Slow inference (<2 tok/s) | a) Raise `--cpus` & `--memory` on the VM, b) lower `-ngl`, c) build with `-O3`. | 
| OOM inside VM | Increase VM memory (`podman machine stop; podman machine rm; podman machine init --memory 16 …`). | 

## 5\. Performance tips

- Start with `-ngl 16` for 7 b, increase until you hit memory limits; each layer ≈ 300 MiB guest RAM.  

- Add `-t 0.7` and `-p "system prompt"` flags for typical chat usage.  

- Prefetch model to a tmpfs inside the VM (`mount -t tmpfs -o size=8G tmpfs /mnt`) to avoid I/O stalls.

With these steps your llama.cpp container will compile against Vulkan, detect the virtio-GPU Venus device, and off-load attention kernels to Apple-Silicon’s GPU through MoltenVK—all while remaining reproducible inside Podman.

Sources