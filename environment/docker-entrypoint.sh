#!/bin/bash
set -e

MODE=${1:-serve}

run_smoke_test() {
    echo "Starting application for smoke test..."
    uvicorn app:app --host 0.0.0.0 --port 8000 &
    APP_PID=$!
    
    cleanup() {
        echo "Stopping application..."
        kill $APP_PID 2>/dev/null || true
    }
    trap cleanup EXIT
    
    echo "Running smoke tests..."
    python smoke_test.py --url http://localhost:8000 --wait
    TEST_RESULT=$?
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo "Smoke tests passed!"
        exit 0
    else
        echo "Smoke tests failed!"
        exit 1
    fi
}

case "$MODE" in
    serve)
        echo "Starting application in serve mode..."
        exec uvicorn app:app --host 0.0.0.0 --port 8000
        ;;
    test)
        run_smoke_test
        ;;
    *)
        echo "Usage: $0 {serve|test}"
        echo "  serve - Start the application server (default)"
        echo "  test  - Run smoke tests against the application"
        exit 1
        ;;
esac
