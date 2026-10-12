"""
Tests for dit.utils.logger.
"""

from loguru import logger

from dit.utils.logger import basic_logger


def test_basic_logger_does_not_disable_user_logging():
    messages = []
    sink = logger.add(messages.append, level="INFO")
    logger.enable("dit")
    try:
        basic_logger("dit.test", None)
        logger.patch(lambda r: r.update(name="dit.test")).info("still visible")
    finally:
        logger.disable("dit")
        logger.remove(sink)
    assert any("still visible" in m for m in messages)
