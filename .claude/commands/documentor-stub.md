---
allowed-tools: *
description: Review a pre-existing stub documentation file for instructions, and initiate codebase analysis for completing the task. 
argument-hint: [stub file]
---

You have been asked to complete documentation that's specified in a stub file; that stub file contains detailed requirements, which you should remove and replace with your outputs when you are done. There may be multiple outcomes specificed, and so you must use the beads task management tool to schedule your subagents.

Use your codebase analysis skill to generate the requested documentation. Use parallel subagents whenever possible - tokens are expensive and we must be thrifty.

Read that file now to recieve the instructions: @$ARGUMENTS.  
