from conclave.llm import HeuristicProvider
from conclave.memory import ConversationBuffer, SummaryMemory, VectorMemory


def test_buffer_keeps_recent_and_recalls_newest_first():
    buf = ConversationBuffer(max_items=3)
    for i in range(5):
        buf.add(f"item {i}")
    assert len(buf) == 3  # only last 3 kept
    recalled = buf.recall("anything", k=2)
    assert recalled == ["item 4", "item 3"]


def test_vector_memory_ranks_by_relevance():
    vm = VectorMemory()
    vm.add("the cat sat on the mat")
    vm.add("python is a programming language")
    vm.add("agents collaborate to solve tasks")
    top = vm.recall("programming language python", k=1)
    assert top and "python" in top[0]


def test_vector_memory_empty():
    assert VectorMemory().recall("x") == []


def test_summary_memory_compresses():
    mem = SummaryMemory(HeuristicProvider(), max_chars=10)
    mem.add("a long piece of content that exceeds the threshold")
    out = mem.recall("anything")
    assert out and isinstance(out[0], str)
