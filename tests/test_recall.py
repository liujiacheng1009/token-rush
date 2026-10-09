"""Keyword recall. No GPU and no chat file unless the user asked about an older chat."""
from tokenrush.recall import asks_about_past, recall


class Store:
    def __init__(self):
        self.reads = 0
        self.chats = [{"id": "old", "title": "旧的学习率笔记", "created": 1760000000,
                       "messages": [{"role": "user", "content": "把学习率定成 1e-4。"}]}]

    def list(self):
        self.reads += 1
        return [{"id": c["id"], "title": c["title"], "created": c["created"]} for c in self.chats]

    def get(self, chat_id):
        self.reads += 1
        for c in self.chats:
            if c["id"] == chat_id:
                return c
        return None


def test_unasked_recall_does_not_read_chats():
    store = Store()
    text = recall(store, "学习率", "new", [{"role": "user", "content": "继续写代码"}])
    assert store.reads == 0 and "未查找" in text
    assert asks_about_past([{"role": "user", "content": "上次说的学习率是多少"}])


def test_conflict_marks_this_chat_as_the_one_to_follow():
    store = Store()
    text = recall(store, "学习率", "new", [
        {"role": "user", "content": "学习率用 3e-4"},
        {"role": "user", "content": "上次说的学习率是多少"},
    ])
    assert store.reads > 0
    assert "旧的学习率笔记" in text and "1e-4" in text and "3e-4" in text and "以本段为准" in text
