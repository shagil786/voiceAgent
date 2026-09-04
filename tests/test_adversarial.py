def test_suite_passes_on_stub_brain():
    import sys
    sys.path.insert(0, "scripts")
    from adversarial import run_suite
    passed, failed = run_suite()
    assert failed == [] and passed == 50
