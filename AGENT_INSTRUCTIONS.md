# Instructions

**Task management**: 
- This project uses `bd` commands instead of markdown TODOs or other methods of task tracking. 
- Only use `bd` when breaking down coding tasks (always use `bd` for coding tasks!), **not** for general reasoning.
- Always create task descriptions in beads, even if they are short.
- When breaking down tasks, identify the code files that will be affected; each task should contain this information, so you can effectively parallelize.
- When creating tasks using `bd`, always identify any tasks that are parallelizeable using subagents. If two tasks will not touch the same code files, they can probablybe parallelized.
**Proposals and specifications**: Feature, implementation, and architecture proposals are located in `features/`
**Project documentation**: 
- Code and architecture documentation for the entire project resides in `docs/`.
- Individual code modules should contain a `README.md` explaining their usage, as a supplement for docstrings.
- All python classes, methods, functions, and modules must have rigorous but concise docstrings.
**Source code**: This is a python library, and its distributable code is located in `src/`.