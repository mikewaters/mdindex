# Python import analysis

Create a python script in @scripts that performs the following: inventories the usage of exports from some library module elsewhere in the library.

## Planning
Before starting implementation, I want you to propose a few plans for how you are going to identify the module symbol usage.

## Parameters
The script should accept a module path as input. For example: "pmd.sources".
When the script is initialized, it should verify that that module exists and can be imported.

## Script Logic
Once initialized and access to the desired module is validated, the script should:
1. Enumerate all of that module's exports, for example via the module's `__all__` attribute
2. For each of the module's export, identify it's fully-qualified name; for example, `pmd.sources` may export a `DocumentSource` symbol, but this symbol is implemented in `pmd.sources.content.base`. This will be part of the output
3. For each export, find all usages *outside of* the module being searched. Using our `DocumentSource` example, find all usages *not* in `pmd.sources`. For each usage, collect the fully qualified path to the object that uses it, using pytest style. An example of this style is `pmd.services.indexing:IndexingService.index_collection`; this points directly to a method that might use the symbol we're looking for. For each option, also collect the parent-most name of the using module, in this case it is `pmd.services`.

## Output style
Once all usages of each symbol in the requested module has been located, output data should be collected.
There should be a dictionary structure of the following format:
```json
{
    "pmd.sources": {
        "DocumentSource": {
            "path": "pmd.sources.content.base.DocumentSource",
            "usage": {
                "pmd.services": [
                    "pmd.services.indexing:IndexingService.index_collection"
                ]
            } 
        }
        "AnotherExport": ...
    }
}
```
In the current iteration, we should putput this data as JSON.

## Future Requirements
Keep in mind that we will eventually pivot to a script that can perform this analysis across an entire repository, generating a complete inventory of cross-module symbol usage within a library.