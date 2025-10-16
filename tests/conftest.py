"""pytest configuration for letta-evals tests"""


def pytest_addoption(parser):
    parser.addoption("--suite-path", action="store", default=None, help="Path to a specific suite.yaml file to test")
