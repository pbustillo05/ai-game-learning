from __future__ import annotations

import signal

import torch

from pbg.core.preemption import PreemptionController


def test_preemption_handler_only_records_notice() -> None:
    controller = PreemptionController()
    controller._handler(signal.SIGTERM, None)
    notice = controller.notice()
    assert notice.requested
    assert notice.signal_number == signal.SIGTERM
    assert notice.signal_name == "SIGTERM"


def test_local_synchronized_notice_preserves_empty_state() -> None:
    controller = PreemptionController()
    notice = controller.synchronized_notice(torch.device("cpu"))
    assert not notice.requested
    assert notice.signal_number is None
