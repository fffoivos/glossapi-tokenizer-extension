from ..bibliography_nextgen_worst_docs_site import line_status


def test_line_status_matrix() -> None:
    assert line_status(predicted=True, truth=True, trusted=True) == "true_positive"
    assert line_status(predicted=True, truth=False, trusted=True) == "false_positive"
    assert line_status(predicted=False, truth=True, trusted=True) == "false_negative"
    assert line_status(predicted=False, truth=False, trusted=True) == "true_negative"
    assert line_status(predicted=True, truth=True, trusted=False) == "untrusted"
