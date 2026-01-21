---
tags:
  - readme 📌
---
# life.md

<https://renormalize.substack.com/p/my-markdown-project-management-system>

```javascript
About this document
===================

## what is this?
- this document is a template for the "nerve center" of a complete project management system. i have been managing my professional and personal tasks with this for at least 5 years. it is Markdown-formatted, but really, all it does is use section headings and sub-headings.

## installation
- copy the text in this file into your favorite text editor and name the file todo-readme.md.
- make a copy of todo-readme.md and rename it todo.md. delete the "About this document" section and all the non-header text, so that only the section headings Landing, Today, etc. are left.
- you are now ready to use the system. refer back to todo-readme.md for instructions.

## usage guidelines
- don't be intimidated. this system can be as simple or complex as you need it to be. a starter configuration could use only the headings Landing, Today, This Week, and Next Week with no auxiliary documents.
- i change the structure to suit my goals as needed. below you will see hints at ways to do this. the flexibility is a specific advantage of the plain text format. however, for me, the system's core elements have remained stable over many years. therefore, i think their purpose should be understood before they are modified or discarded.
- keep this document lean and focused on prioritization. ideally, like a kanban, this document should be like a trusted secretary who is always ready to answer the question "here are the three most important things you could be doing right now." if you let the document's top sections scope-creep to encompass all your other forms of ideation, it will get too long and complicated, you'll begin to dislike and avoid it, and it will no longer help you. move peripheral things out to other documents (peripheral doesn't mean "not important", but "not in immediate need of the sharpest focus"). we'll point at ways to do that below.

## why use sublime?
i format the list using markdown and edit in the Sublime Text editor using a dark Monokai theme. reasons: 
- extremely low latency. work at the speed of thought.
- colored markdown section headings. red = level 1, yellow = level 2.
- it's a tabbed editor with has an optional directory tree sidebar. this is extremely useful because we often want to have a few active working documents and quickly move text from one document to another.
- it's full of thoughtful little ergonomic features, such as auto-indented soft wrap of multi-line bullets on indented lists.

## auxiliary files
- log.md: for documenting the achievement of notable milestones. entries in this log are Markdown sections, titled with the date and a brief topic description. i default to reverse chronological order, so text is added conveniently at the top.
- dump.md: for notes that no longer have any obvious use but you have a hard time throwing away. this is really just a paste bin to help you let go of things. don't worry too much about structuring it. just paste new stuff in at the top.
- done.md: for tracking completed tasks. when a task is complete, you can paste it in at the top of done.md. this can be helpful if you sometimes need to remember whether and when tasks were done. for example, if someone were to lose information you provided them, and you could consult done.md to help you recall what you provided and when. modify the 

## other aspects of the system not shown here
- an "todo" directory tree comprising the entire personal project management system. the list here is only the COO's office, so to speak. it's not even the CEO. the really visionary stuff happens in a more freeform way.
- subdirectories for projects, so they can have multiple files.
- meeting notes, which are markdown formatted like this doc and have filenames like "YYYY-MM-DD meeting topic . md"
- year subfolders (2023/, 2024/, etc) used primarily for assorted meeting notes and little things that i want to save but don't care too much about where they go. (you can usually find with keyword search anyway)

## what this system does not solve for
- this is a personal system. i make no recommendations for how to use it, if at all, when collaborating with others. however, i believe every individual should be the leader of their own life. to that end, there is great value in having a personal system that says exactly what you want it to say, without regard to the opinions of others.
- i speculate that using this system with a highly responsive collaborative text editor like Google Docs could make it viable for collaborations, but have not tested this.


Daily Reminders
===============
- this section is for reminding yourself about the sort of things you need to be constantly vigilant about in yourself. e.g. the mistakes you tend to repeatedly make and want to improve on.
- example: i recently decided to add Rumi's three gates of speech here. before saying something, ask yourself: "is it true? is it necessary? is it kind?" if you sometimes run your mouth and end up regretting it, rumi's gates are a welcome antidote.


Landing
=======

- when you learn about something you want or need to track the completion of, put it in this section immediately. the section can also be called Triage, but Landing is a bit nicer-sounding.
- the point of this section is to keep you from worrying about where the task should go. we need to separate recording from organization. they can't be done at the same time, it's often too cognitively burdensome.


Today
=====

- small things you want to do today
- more things

## project Acorn
ref: "project Acorn.md"
7/23 - deliver XYZ
- this subsection represents a personal or work project, tracks all info related to the project.
- when the time sensitivity of a project changes, move the section as a block to the most appropriate time heading using cut-and-paste.
- you can have as many projects as you want, and record whatever info needs recording here.
- if a project's notes get too long, make a separate planning document for the project. optional: "link" the document with a "ref" to its location, as above. the design of separate project documents is generally roadmap-oriented. keeping track of time-sensitive stuff happens here, because if you spread it too much you'll lose track.
- larger projects may eventually require their own directories, in which case the ref should reflect the directory, and typically the full path to the main project planning document.


This Week
=========

- things you want to do this week

## project B
ref: project-B
- project B is not an immediate focus right now, but it's up soon, so it's in the "this week" bucket


Next Week
=========

- things you want to do next week
- you can add more time periods, like months, quarters, or years, to give space to "important but not urgent" tasks. this helps you stay conscious of Quadrant 2 in Covey's importance-urgency matrix.


Backlog 1
=========
- no specific timeframe for these tasks
- things you'd like to get around to, but you're not hard-committed to them


Backlog 2
=========
- stuff that's lower priority than Backlog 1
- helps keep Backlog 1 clean
- there's probably a better name for these than Backlog 1 and 2


Done
====
- This can be useful for tracking tasks that have been completed, but you're not quite ready to delete your records of them yet. for example, you may need evidence later that you did something to convince yourself you don't still need to do it. however, you will likely prefer using done.md for this if you use a fast tabbed editor with a directory tree sidebar.
```