---
tags:
  - type
objectType: "[Workflow.md](./Workflow.md)"
domain scope:
  - Information
---
# Reference Management

Is a:: Workflow

```plain
class ReferenceManager:

  def save_url(source: URL):
    ""
    Dont know what to do with this, and dont
    want to make a decision now
    """
    # I shouldnt be abusing my readitlater app in this way
    ContentReader.save(source)

    # instead, i should do this:
    ContentBot.figure_out(source)

  def bookmark(source: URL, 
              thread: Optional[Thread],
              tags: Optional[dict]):
    """
    We aren't going to read this now,
    but we expect to need it later and so
    it should be saved - with as much 
    context as possibkle - and be snapshotted/
    back-up to ensure we don't lose it.
    `ReferenceStore` may log this action,
    if its deemed significant, which adds
    more context to the full picture.
    ""
    ReferenceStore.save(source, 
        folder=thread,
        tags=tags,
        snapshot=True)

  def readlater(source: URL, tags: Optional[dict]):
    """
    I want to read this later - like **want** want -
    and next time i go to read something it should
    be available to me.
    ""
    ContentReader.save(source,
      tags=tags)
```

Managing resources and information that I come across, so that it arrives in various systems in the platform and that it is classified in the best way possible (to minimize the friction of picking it up later).

## Components

[Reference Store.md](./Reference%20Store.md)

[Content Reader](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/53f4fb80-346d-440f-ac24-2d8a636ae578)

[Browser](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/ad938437-c8c7-47d5-a2c9-68d177b721be)

[Sync Cache](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/d122ac4f-d9fe-4433-a200-d7c64aae3907)

[Second Brain](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/57b06079-3bbf-4b78-aae1-8876c2cd97aa)

[Pod Catcher.md](./Pod%20Catcher.md)

## Functions

- [Highlight Sync](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/20c9f7af-5e93-4aaf-9c22-cd8dfcaaf965)

- [Tab Group Exporter](https://app.heptabase.com/2f7caf87-d999-4778-8e30-61689601271e/card/2572cb1d-df82-4fb6-841c-889ff8337d11)

---

Activity diagram ([Miro](https://miro.com/app/board/uXjVNYd7VI0=/?moveToWidget=3458764574128661004&cot=14)):



![image.png](./Reference%20Management-assets/image.png)