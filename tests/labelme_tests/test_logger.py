import logging

from labelme.logger import ColoredFormatter


def _record():
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=12,
        msg="value=%r",
        args=(123,),
        exc_info=None,
    )


def test_colored_formatter_interpolates_arguments():
    formatter = ColoredFormatter("%(message2)s", use_color=True)

    assert "value=123" in formatter.format(_record())


def test_uncolored_formatter_populates_custom_fields():
    formatter = ColoredFormatter(
        "%(levelname2)s %(module2)s:%(lineno2)s %(message2)s",
        use_color=False,
    )

    formatted = formatter.format(_record())

    assert "INFO" in formatted
    assert "value=123" in formatted
