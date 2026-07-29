from pysbagen.sleep_cli import collect_sleep_request


def test_conversation_collects_human_sleep_request():
    answers = iter(["1", "2", "3", "2"])
    output = []
    request = collect_sleep_request(input_fn=lambda prompt: next(answers), print_fn=output.append)
    assert request.problem == "racing_mind"
    assert request.sound_world == "slow_night_music"
    assert request.intensity == "immersive"
    assert request.duration_minutes == 45
    assert any("do not need to know anything about frequencies" in line for line in output)
