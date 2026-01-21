---
tags:
  - document 📑
---
# Assembling a Strong Password 

### Using the CLI

One can use the builtin dictionary to find words for s good password.

#### Finding words that start with a `t`:

`grep -E "^t" /usr/share/dict/words`

#### Getting random words:

`shuf -n4 /usr/share/dict/words`

Assembling the words into a single string (hyphen delimited):

`shuf -n4 /usr/share/dict/words | paste -s -d "-" -`