from arcana_ml import ping

def test_ping() -> None:
    assert ping() == "fingerprint-py"
