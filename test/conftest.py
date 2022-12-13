# Copyright 2022 Johannes Reiff
# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_addoption(parser):
    parser.addoption('--skip-slow',
        action='store_true', default=False, help='skip slow tests')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--skip-slow'):
        skip_slow = pytest.mark.skip(reason='skipped due to --skip-slow')
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)
