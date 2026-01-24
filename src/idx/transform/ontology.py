from llama_index.core.schema import TransformComponent

NOTE_TYPE_MAP = {
    "book-note": "NoteType.BOOK_SUMMARY",
    "meeting": "NoteType.MEETING_NOTE",
}

TOPIC_MAP = {
    "rag": "Topic.RAG",
    "llamaindex": "Topic.LLAMA_INDEX",
}

class OntologyMapper(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            m = node.metadata

            raw_type = m.get("type")
            if raw_type:
                m["note_type"] = NOTE_TYPE_MAP.get(raw_type, "NoteType.UNKNOWN")

            tags = m.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]

            ont_topics = [
                TOPIC_MAP[t] for t in tags if t in TOPIC_MAP
            ]
            m["topics"] = ont_topics

            # Keep original frontmatter under a namespaced key if needed
            # m["obsidian_frontmatter"] = { ... }

        return nodes
