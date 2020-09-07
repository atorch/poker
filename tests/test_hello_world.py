from poker import hello_world


def test_say_hello():

    phrase = hello_world.say_hello()

    # This test should fail
    assert phrase == "I know how to play poker"
