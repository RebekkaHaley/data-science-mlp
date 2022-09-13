import pytest

from data_science_mlp.rgb import InputError, RGB, generate_RGB_data


def test_input_error() -> None:
    error = InputError('foobar', 'testing')
    assert error.expr == 'foobar'
    assert error.msg == 'testing'


def test_rgb_valid() -> None:
    dummy_colour = (11, 22, 33)
    dummy_rgb = RGB(dummy_colour[0], dummy_colour[1], dummy_colour[2])
    assert dummy_rgb.R == dummy_colour[0]
    assert dummy_rgb.G == dummy_colour[1]
    assert dummy_rgb.B == dummy_colour[2]
    assert dummy_rgb.RGB == dummy_colour
    assert dummy_rgb.hex == '#{:02X}{:02X}{:02X}'.format(dummy_rgb.R, dummy_rgb.G, dummy_rgb.B)

def test_rgb_invalid_over() -> None:
    with pytest.raises(InputError):
        dummy_colour = (11, 22, 99999)
        RGB(dummy_colour[0], dummy_colour[1], dummy_colour[2])


def test_rgb_invalid_under() -> None:
    with pytest.raises(InputError):
        dummy_colour = (11, 22, -99999)
        RGB(dummy_colour[0], dummy_colour[1], dummy_colour[2])


def test_generate_rbg_data() -> None:
    rgb_X = 5
    rgb_extreme = True
    rgb_data = generate_RGB_data(X=rgb_X, extreme=rgb_extreme)
    assert type(rgb_data[0].RGB) == tuple
    assert len(rgb_data) == rgb_X
