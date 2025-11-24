# running tests

all tests run in docker containers to ensure consistent environment

## quick start

```bash
# run all tests with coverage
make test

# run only unit tests
make test-unit

# run only integration tests
make test-integration

# run tests and generate html coverage report
make test-coverage
```

## docker commands

if you prefer docker-compose directly:

```bash
# run all tests
docker-compose up --build test

# run specific test file
docker-compose run --rm test pytest tests/unit/test_ingest.py -v

# run specific test class
docker-compose run --rm test pytest tests/unit/test_ingest.py::TestDownloadOne -v

# run with more verbose output
docker-compose run --rm test pytest tests/ -vv
```

## test structure

```
tests/
├── conftest.py              # shared fixtures for all tests
├── unit/                    # unit tests for individual functions
│   ├── test_ingest.py      # 15 tests for ingest service
│   ├── test_features.py    # 26 tests for features service
│   └── test_schemas.py     # 15 tests for pandera schemas
└── integration/             # integration tests for full pipelines
    ├── test_ingest_pipeline.py    # 3 tests for ingest main()
    └── test_features_pipeline.py  # 7 tests for features main()
```

## coverage

current coverage: **98%**
- 66 tests passing
- only 3 lines missing coverage (if __name__ == "__main__" blocks)

## docker image

tests use dedicated `vf-test` image that includes:
- all base dependencies (pandas numpy pyarrow pandera pydantic)
- ingest dependencies (yfinance)
- test dependencies (pytest pytest-cov pytest-mock pytest-xdist)

image is built from `Dockerfile.test` which extends `vf-base`
