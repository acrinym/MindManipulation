import os
from pysbagen.parser import parse_sbg

def test_parse_sbg():
    # Create a dummy .sbg file for testing
    sbg_content = """
    # This is a test
    alpha: 200+10/50
    beta: 300+20/60

    NOW alpha
    0:10 beta
    """
    sbg_path = "test.sbg"
    with open(sbg_path, "w") as f:
        f.write(sbg_content)

    tone_sets, schedule = parse_sbg(sbg_path)

    assert "alpha" in tone_sets
    assert "beta" in tone_sets
    assert len(schedule) == 2
    assert schedule[0][0] == 0
    assert schedule[1][0] == 10

    os.remove(sbg_path)
