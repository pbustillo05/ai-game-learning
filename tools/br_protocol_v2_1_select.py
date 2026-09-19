#!/usr/bin/env python3
"""Promote and freeze protocol-v2.1 candidates without reading test data."""

import br_protocol_v2_select as selector


if __name__ == "__main__":
    selector.PROTOCOL = "fixed_target_paired_response_gain_v2_1"
    selector.REPORT_SUFFIX = "v2_1"
    selector.SCHEMA_VERSION = "v2.1"
    selector.main()
