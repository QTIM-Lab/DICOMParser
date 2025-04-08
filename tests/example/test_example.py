import pytest


@pytest.mark.parametrize("num1, num2, expected", [
    (2, 3, 6),
    (5, 4, 20),
    (10, 0, 0),
    (-2, 3, -6),
    (2.5, 2, 5.0),
])
def test_multiplication(num1, num2, expected):
    """
    Test the multiplication of two numbers.
    """
    # Act
    result = num1 * num2

    # Assert
    assert result == expected