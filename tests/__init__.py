import wdbx as wdbx

def test_wdbx():
    # Initialize the WDBX environment
    db = wdbx.WDBX().__init__()

    # Test basic functionality
    assert db.is_connected()

    # Test data operations
    test_data = {"key": "value"}
    db.insert(test_data)
    result = db.query({"key": "value"})
    assert result == test_data

    # Clean up
    db.close()
