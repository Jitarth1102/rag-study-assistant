import uuid

from rag_assistant.vectorstore.point_id import make_point_uuid


def test_point_uuid_deterministic():
    identity = "s1:a1:1:0"
    pid1 = make_point_uuid(identity)
    pid2 = make_point_uuid(identity)
    assert pid1 == pid2
    uuid_obj = uuid.UUID(pid1)
    assert str(uuid_obj) == pid1
