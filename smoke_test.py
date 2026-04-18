import os
import sys
import httpx
import time
import argparse

SMOKE_TEST_TIMEOUT = 60
HEALTH_CHECK_RETRIES = 10
HEALTH_CHECK_DELAY = 2


def wait_for_server(base_url: str) -> bool:
    for i in range(HEALTH_CHECK_RETRIES):
        try:
            response = httpx.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✓ Server is ready (attempt {i + 1})")
                return True
        except httpx.RequestError:
            pass
        print(f"Waiting for server... (attempt {i + 1}/{HEALTH_CHECK_RETRIES})")
        time.sleep(HEALTH_CHECK_DELAY)
    return False


def run_smoke_tests(base_url: str) -> bool:
    all_passed = True
    
    print("\n" + "=" * 50)
    print("Running Smoke Tests")
    print("=" * 50)
    
    print("\n[1/5] Health Check...")
    try:
        response = httpx.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed: {data}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        all_passed = False
    
    print("\n[2/5] Model Status (before training)...")
    try:
        response = httpx.get(f"{base_url}/model/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model status: model_exists={data.get('model_exists')}")
        else:
            print(f"✗ Model status failed: {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"✗ Model status error: {e}")
        all_passed = False
    
    print("\n[3/5] Model Training...")
    try:
        response = httpx.post(
            f"{base_url}/train",
            json={"n_estimators": 10, "max_depth": 5},
            timeout=SMOKE_TEST_TIMEOUT
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Training completed: accuracy={data.get('evaluation', {}).get('accuracy', 'N/A')}")
        else:
            print(f"✗ Training failed: {response.status_code} - {response.text}")
            all_passed = False
    except Exception as e:
        print(f"✗ Training error: {e}")
        all_passed = False
    
    print("\n[4/5] Prediction...")
    try:
        response = httpx.post(
            f"{base_url}/predict",
            json={
                "feature_1": 1.0,
                "feature_2": 2.0,
                "feature_3": 3.0,
                "feature_4": 4.0,
                "feature_5": 5.0
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Prediction: class={data.get('prediction')}, probabilities={data.get('probabilities')}")
        else:
            print(f"✗ Prediction failed: {response.status_code} - {response.text}")
            all_passed = False
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        all_passed = False
    
    print("\n[5/5] Metrics...")
    try:
        response = httpx.get(f"{base_url}/metrics", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Metrics retrieved: accuracy={data.get('accuracy')}, precision={data.get('precision')}")
        else:
            print(f"✗ Metrics failed: {response.status_code}")
            all_passed = False
    except Exception as e:
        print(f"✗ Metrics error: {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All smoke tests PASSED")
    else:
        print("✗ Some smoke tests FAILED")
    print("=" * 50)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Smoke tests for AI-ML-Demo")
    parser.add_argument(
        "--url",
        default=os.environ.get("APP_URL", "http://localhost:8000"),
        help="Base URL of the application"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for server to be ready before running tests"
    )
    args = parser.parse_args()
    
    if args.wait:
        print(f"Waiting for server at {args.url}...")
        if not wait_for_server(args.url):
            print("✗ Server did not become ready in time")
            sys.exit(1)
    
    success = run_smoke_tests(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
