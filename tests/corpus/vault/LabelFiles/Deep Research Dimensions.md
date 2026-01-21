# Deep Research Dimensions

When Deep Research [LifeOS Agents.md](./LifeOS%20Agents.md) are provided a Resource to analyze, they need to:

1. Determine which dimensions to capture from the resource, by classifying the Resource in a special way - these are pretty much any Topic in my Interests, as well as any child topics therein, and any other documents where it is mentioned

2. Figure out where to put the analysis results

The dimensions are important because for each resource it needs to understand what properties of it are desired in the analysis. An example of this would be let's say there's a Git repository. It's a Python project and it's related to in-context learning. First and foremost we want to know about that Python project: what's the license, what packaging does it use, does it sit on top of something like FastAPI or Pydantic AI? What major frameworks does it use, does it require any external observability tools that need to be running, what kind of database does it use if any? Does it use an ORM? Blah blah blah.

In order to get these dimensions, it needs to first classify the resource within its ontology of the kind of thing it is. So what this means is it needs to get to that list of required dimensions somehow. And it does that by classifying what the thing is or asking the user for instructions.

With respect to where to put the analysis, there may be an existing document in the system that has a landscape purpose. So, you know, I might have a Python GUI tools landscape collection. And you know, what I throw a link for a GitHub repository up, the agent should be smart enough to associate that with that that data information collection effort, the landscape document, and then include the analysis results in that document both on disk in like this, the collection itself wherever it is, but also within the substrate. So that other consumers of the substrate know that this other this new Python thing was researched and associated with a project and the landscape and anything else that stems from the project like a goal or whatever. An example of this is [Python UI Landscape.md](./Python%20UI%20Landscape.md) 

### Associating a resource with topics to determine which dimensions to capture during analysis

In a hierarchical list of topics, one or more of these topics in the tree can be associated as a dimension for analysis. When a resource is examined to get the list of dimensions that should be looked at in doing deep research for that resource, it should use the topic hierarchy as the taxonomy for category identification of the thing.



So what this means is there needs to be a parallel hierarchy or data structure to the topic taxonomies that identify which topical entities are dimensions. Importantly, if a given resource is found to be categorized under that dimension, the system should do is walk up the topic hierarchy looking for other parent topics that also are dimensions and if they are then include them in the dimensional analysis.



So for each of these dimensions there will be a set of instructions for what information to capture, why it's meaningful, and what to do with it. And so if multiple topics in a single branch of taxonomy tree are matched, then it would use the combination of those instructions from each of those. That doesn't mean that there aren't other branches of that taxonomy or other taxonomies that would also be matching categories for the resource. So the result would be all of those instructions are ended together to give to the deep research agent. 