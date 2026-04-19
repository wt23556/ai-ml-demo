import os
import sys
import time
import json
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Any

BASE_URL = "http://localhost:8088"

class ModelTestSuite:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def log_result(self, test_name: str, status: str, details: Any = None, duration: float = 0):
        result = {
            "test_name": test_name,
            "status": status,
            "duration": round(duration, 4),
            "details": details
        }
        self.results.append(result)
        if status == "PASS":
            self.passed += 1
        else:
            self.failed += 1
        
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {test_name} [{duration:.4f}s]")
        if details and status == "FAIL":
            print(f"   详情: {details}")

    def test_health_endpoint(self):
        start = time.time()
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=10)
            if response.status_code == 200 and response.json().get("status") == "healthy":
                self.log_result("健康检查接口", "PASS", response.json(), time.time() - start)
            else:
                self.log_result("健康检查接口", "FAIL", f"状态码: {response.status_code}", time.time() - start)
        except Exception as e:
            self.log_result("健康检查接口", "FAIL", str(e), time.time() - start)

    def test_boundary_values_prediction(self):
        test_cases = [
            ("极小值", {"feature_1": 0.001, "feature_2": 0.001, "feature_3": 0.001, "feature_4": 0.001, "feature_5": 0.001}),
            ("极大值", {"feature_1": 1000.0, "feature_2": 1000.0, "feature_3": 1000.0, "feature_4": 1000.0, "feature_5": 1000.0}),
            ("负值输入", {"feature_1": -1.0, "feature_2": -5.5, "feature_3": -10.0, "feature_4": -0.5, "feature_5": -2.0}),
            ("零值输入", {"feature_1": 0, "feature_2": 0, "feature_3": 0, "feature_4": 0, "feature_5": 0}),
            ("典型鸢尾花0类", {"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2, "feature_5": 0.3}),
            ("典型鸢尾花1类", {"feature_1": 7.0, "feature_2": 3.2, "feature_3": 4.7, "feature_4": 1.4, "feature_5": 1.5}),
            ("典型鸢尾花2类", {"feature_1": 6.3, "feature_2": 3.3, "feature_3": 6.0, "feature_4": 2.5, "feature_5": 2.5}),
        ]
        
        predictions = []
        all_pass = True
        start = time.time()
        
        for case_name, features in test_cases:
            try:
                response = requests.post(f"{BASE_URL}/predict", json=features, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    predictions.append({
                        "case": case_name,
                        "prediction": data["prediction"],
                        "probabilities": [round(p, 4) for p in data["probabilities"]]
                    })
                else:
                    all_pass = False
            except Exception as e:
                all_pass = False
        
        status = "PASS" if all_pass else "FAIL"
        self.log_result("边界值预测测试", status, predictions, time.time() - start)
        if all_pass:
            print("   预测详情:")
            for p in predictions:
                print(f"     - {p['case']}: 类别={p['prediction']}, 概率={p['probabilities']}")

    def test_error_handling(self):
        test_cases = [
            ("缺少字段", {}),
            ("字段类型错误", {"feature_1": "not_a_number", "feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2, "feature_5": 0.3}),
            ("缺少feature_1", {"feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2, "feature_5": 0.3}),
            ("多余字段", {"feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, "feature_4": 0.2, "feature_5": 0.3, "extra": "value"}),
        ]
        
        error_responses = []
        start = time.time()
        
        for case_name, payload in test_cases:
            try:
                response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
                error_responses.append({
                    "case": case_name,
                    "status_code": response.status_code,
                    "has_error": response.status_code >= 400
                })
            except Exception as e:
                error_responses.append({
                    "case": case_name,
                    "error": str(e)
                })
        
        proper_error_handling = all(r.get("has_error", False) for r in error_responses if "status_code" in r)
        status = "PASS" if proper_error_handling else "FAIL"
        self.log_result("错误处理测试", status, error_responses, time.time() - start)

    def test_training_different_parameters(self):
        param_combinations = [
            ("n_estimators=10", {"n_estimators": 10}),
            ("n_estimators=50", {"n_estimators": 50}),
            ("n_estimators=200", {"n_estimators": 200}),
            ("max_depth=2", {"n_estimators": 100, "max_depth": 2}),
            ("max_depth=10", {"n_estimators": 100, "max_depth": 10}),
            ("默认参数", {}),
        ]
        
        training_results = []
        start = time.time()
        
        for param_name, params in param_combinations:
            try:
                train_start = time.time()
                response = requests.post(f"{BASE_URL}/train", json=params, timeout=30)
                train_time = time.time() - train_start
                
                if response.status_code == 200:
                    data = response.json()
                    training_results.append({
                        "parameters": param_name,
                        "accuracy": round(data["evaluation"]["accuracy"], 6),
                        "training_time": round(train_time, 4),
                        "n_estimators": data["training"]["n_estimators"],
                        "max_depth": data["training"]["max_depth"]
                    })
                else:
                    training_results.append({
                        "parameters": param_name,
                        "error": f"HTTP {response.status_code}"
                    })
            except Exception as e:
                training_results.append({
                    "parameters": param_name,
                    "error": str(e)
                })
        
        all_success = all("error" not in r for r in training_results)
        status = "PASS" if all_success else "FAIL"
        self.log_result("多参数训练测试", status, training_results, time.time() - start)
        
        if all_success:
            print("   训练结果对比:")
            df = pd.DataFrame(training_results)
            print(df.to_string(index=False))

    def test_batch_prediction_performance(self):
        np.random.seed(42)
        n_requests = 50
        
        predictions = []
        start = time.time()
        
        for i in range(n_requests):
            features = {
                "feature_1": np.random.uniform(4, 8),
                "feature_2": np.random.uniform(2, 5),
                "feature_3": np.random.uniform(1, 7),
                "feature_4": np.random.uniform(0, 3),
                "feature_5": np.random.uniform(0, 3)
            }
            try:
                response = requests.post(f"{BASE_URL}/predict", json=features, timeout=10)
                predictions.append(response.status_code == 200)
            except Exception:
                predictions.append(False)
        
        total_time = time.time() - start
        success_rate = sum(predictions) / len(predictions)
        avg_time = total_time / n_requests
        
        details = {
            "total_requests": n_requests,
            "successful_requests": sum(predictions),
            "success_rate": f"{success_rate * 100:.2f}%",
            "total_time": round(total_time, 4),
            "avg_time_per_request": round(avg_time * 1000, 2)
        }
        
        status = "PASS" if success_rate >= 0.95 else "FAIL"
        self.log_result("批量预测性能测试", status, details, total_time)
        
        print(f"   {n_requests}次预测，成功率: {details['success_rate']}, 平均耗时: {details['avg_time_per_request']}ms")

    def test_concurrent_predictions(self):
        import concurrent.futures
        import threading
        
        def make_prediction():
            features = {
                "feature_1": 5.1, "feature_2": 3.5, "feature_3": 1.4, 
                "feature_4": 0.2, "feature_5": 0.3
            }
            try:
                response = requests.post(f"{BASE_URL}/predict", json=features, timeout=10)
                return response.status_code == 200
            except Exception:
                return False
        
        n_concurrent = 20
        start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction) for _ in range(n_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start
        success_rate = sum(results) / len(results)
        
        details = {
            "concurrent_requests": n_concurrent,
            "success_rate": f"{success_rate * 100:.2f}%",
            "total_time": round(total_time, 4)
        }
        
        status = "PASS" if success_rate >= 0.9 else "FAIL"
        self.log_result("并发预测测试", status, details, total_time)

    def test_model_persistence(self):
        start = time.time()
        
        model_path = "/Users/tomtang/ai_test/ai-ml-demo/model/random_forest_model.joblib"
        file_exists_before = os.path.exists(model_path)
        file_size_before = os.path.getsize(model_path) if file_exists_before else 0
        
        try:
            response = requests.post(f"{BASE_URL}/train", json={"n_estimators": 100}, timeout=30)
            
            if response.status_code == 200:
                file_exists_after = os.path.exists(model_path)
                file_size_after = os.path.getsize(model_path) if file_exists_after else 0
                
                details = {
                    "model_file_exists": file_exists_after,
                    "file_size_kb": round(file_size_after / 1024, 2),
                    "file_updated": file_size_before != file_size_after
                }
                
                status = "PASS" if file_exists_after and file_size_after > 0 else "FAIL"
                self.log_result("模型持久化测试", status, details, time.time() - start)
            else:
                self.log_result("模型持久化测试", "FAIL", "训练失败", time.time() - start)
        except Exception as e:
            self.log_result("模型持久化测试", "FAIL", str(e), time.time() - start)

    def test_metrics_and_status(self):
        start = time.time()
        all_pass = True
        details = {}
        
        try:
            metrics_response = requests.get(f"{BASE_URL}/metrics", timeout=10)
            details["metrics_status"] = metrics_response.status_code
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                details["metrics"] = metrics
                assert "accuracy" in metrics
                assert "precision" in metrics
                assert "recall" in metrics
            else:
                all_pass = False
            
            status_response = requests.get(f"{BASE_URL}/model/status", timeout=10)
            details["status_status"] = status_response.status_code
            if status_response.status_code == 200:
                status_data = status_response.json()
                details["model_status"] = status_data
                assert "model_exists" in status_data
                assert status_data["model_exists"] == True
            else:
                all_pass = False
            
            status = "PASS" if all_pass else "FAIL"
            self.log_result("指标和状态接口测试", status, details, time.time() - start)
        except Exception as e:
            self.log_result("指标和状态接口测试", "FAIL", str(e), time.time() - start)

    def test_static_files(self):
        start = time.time()
        files_to_check = [
            "/static/confusion_matrix.png",
            "/static/accuracy_curve.png"
        ]
        
        results = {}
        all_found = True
        
        for file_path in files_to_check:
            try:
                response = requests.get(f"{BASE_URL}{file_path}", timeout=10)
                results[file_path] = {
                    "status_code": response.status_code,
                    "size_bytes": len(response.content)
                }
                if response.status_code != 200 or len(response.content) < 100:
                    all_found = False
            except Exception as e:
                results[file_path] = {"error": str(e)}
                all_found = False
        
        status = "PASS" if all_found else "FAIL"
        self.log_result("静态文件(可视化图表)测试", status, results, time.time() - start)

    def generate_report(self):
        print("\n" + "="*60)
        print("📊 模型测试套件完整报告")
        print("="*60)
        print(f"总计: {len(self.results)} 个测试用例")
        print(f"通过: {self.passed} 个 ✅")
        print(f"失败: {self.failed} 个 ❌")
        print(f"通过率: {(self.passed/len(self.results)*100):.1f}%")
        print("="*60)
        
        print("\n📋 详细测试结果:")
        for r in self.results:
            icon = "✅" if r["status"] == "PASS" else "❌"
            print(f"{icon} {r['test_name']} - {r['duration']}s")
        
        print("\n" + "="*60)
        print("💡 测试总结:")
        print("="*60)
        
        if self.failed == 0:
            print("🎉 所有测试通过！模型功能完整，性能稳定。")
        else:
            print(f"⚠️  有 {self.failed} 个测试失败，建议检查失败项。")
        
        with open("test_report.json", "w") as f:
            json.dump({
                "summary": {
                    "total": len(self.results),
                    "passed": self.passed,
                    "failed": self.failed,
                    "pass_rate": self.passed/len(self.results)
                },
                "details": self.results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存到: test_report.json")


def main():
    print("🚀 开始运行全面的模型测试套件...\n")
    
    suite = ModelTestSuite()
    
    suite.test_health_endpoint()
    suite.test_boundary_values_prediction()
    suite.test_error_handling()
    suite.test_training_different_parameters()
    suite.test_batch_prediction_performance()
    suite.test_concurrent_predictions()
    suite.test_model_persistence()
    suite.test_metrics_and_status()
    suite.test_static_files()
    
    suite.generate_report()


if __name__ == "__main__":
    main()
