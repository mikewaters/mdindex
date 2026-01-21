---
tags:
  - document 📑
---
# Docker with Colima

Using [Colima](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/e854d982-1076-42d2-8a9f-3fa1440e6b6d#585610aa-155a-4a7e-b984-fffa7c3f2b62) as a container engine, which exposes a unix socket compatible with docker tools.

### Install

```python
#!gist

brew install docker
brew install docker-compose
# ... configure docker for compose, and fixup some stuff

brew install qemu
brew install colima

# edit the default VM to expose an ip address
# (a-la -—network-address)
colima template  # fix it ...

colima start # will use qemu
# OR, use MacOS virtualization
colima start --vm-type=vz --vz-rosetta
```

Had to add this to `~/.zshrc`:

```python
# Allow docker-ecosystem tools to connect to the Colima engine
export DOCKER_HOST=$( colima status -j |jq ".docker_socket" |tr -d '"')
```

Using `[lazydocker](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/e854d982-1076-42d2-8a9f-3fa1440e6b6d#c8c0e9d4-afdd-41ae-90e3-f34a11c1298e)` to view and manage containers (which needs the above change).


