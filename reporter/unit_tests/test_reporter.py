import time
from unittest.mock import patch

from reporter.reporting import Reporter


def test_report_logs_one_line_per_frame(make_result):
    with patch("reporter.reporting.log") as mock_log:
        Reporter().report(make_result(n_frames=3))
    assert mock_log.info.call_count == 3


def test_report_logs_single_frame(make_result):
    with patch("reporter.reporting.log") as mock_log:
        Reporter().report(make_result(n_frames=1))
    assert mock_log.info.call_count == 1


def test_report_includes_camera_id_in_log_args(make_result):
    with patch("reporter.reporting.log") as mock_log:
        Reporter().report(make_result(n_frames=1))
    assert "cam_01" in mock_log.info.call_args[0]


def test_report_includes_frame_number_in_log_args(make_result):
    result = make_result(n_frames=1)
    result.frames_meta[0].frame_number = 99
    with patch("reporter.reporting.log") as mock_log:
        Reporter().report(result)
    assert 99 in mock_log.info.call_args[0]


def test_report_e2e_latency_is_positive(make_result):
    result = make_result(n_frames=1)
    result.frames_meta[0].captured_at = time.monotonic() - 0.5  # 500ms ago
    logged_args = []
    with patch("reporter.reporting.log") as mock_log:
        mock_log.info.side_effect = lambda fmt, *args: logged_args.extend(args)
        Reporter().report(result)
    # args: camera_id, frame_number, captured_at, proc_latency_ms, e2e_ms, batch_id
    assert logged_args[4] > 0
