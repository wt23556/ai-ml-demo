## 项目背景

开发一个轻量级 AI 分类/预测模型应用，完成数据预处理、模型训练、评估推理与简单 Web 部署。模型需可训练、可保存、可加载推理，适合作为企业/教学级 AI 演示项目。

## 功能需求

### 1. 数据集加载与预处理
- 从 `data/` 目录加载 CSV 数据集
- 实现缺失值填充、特征归一化
- 自动划分训练集与测试集（80/20）

### 2. 模型训练
- 支持使用 Scikit-learn 的随机森林分类器
- 训练过程输出日志，显示关键参数和进度

### 3. 模型评估
- 计算并输出准确率、精确率、召回率
- 生成混淆矩阵并保存为图片
- 绘制训练过程中的准确率曲线并保存

### 4. 模型保存与加载
- 使用 `joblib` 将训练好的模型保存到 `model/` 目录
- 支持从文件加载模型进行推理

### 5. Web API 接口
- 使用 FastAPI 提供 `/train` 接口触发模型训练
- 提供 `/predict` 接口接收 JSON 数据并返回预测结果
- 提供简单的前端页面，支持可视化操作

### 6. 结果可视化
- 自动生成混淆矩阵和准确率曲线图片
- 在前端页面展示评估结果和图表

## 约束类型

### 技术栈/依赖约束
- 必须使用 Python 3.10+
- 必须使用 Scikit-learn 进行模型训练和评估
- 必须使用 FastAPI 构建 Web 服务
- 必须使用 Pandas 进行数据处理
- 必须使用 Matplotlib/Seaborn 进行可视化
- 禁止使用闭源模型服务，必须本地可运行

### 代码改动约束
- 严格遵循最小改动原则，仅修改需求相关文件
- 禁止删除数据文件、配置文件和已有目录结构

### 交付约束
- 交付代码必须完整可运行，附带完整依赖清单
- 交付前必须完成自测，确保功能符合需求

## 技术栈要求

Python 3.10+、Scikit-learn、Pandas、Matplotlib、FastAPI、Uvicorn

## 项目结构

```
ai-ml-demo/
├── app.py                    # FastAPI 主应用
├── data/
│   └── dataset.csv           # 数据集
├── model/                    # 模型存储目录
├── src/
│   ├── __init__.py
│   ├── data_processor.py     # 数据处理模块
│   ├── model_trainer.py      # 模型训练模块
│   └── model_evaluator.py    # 模型评估模块
├── static/                   # 静态文件（图表）
├── templates/
│   └── index.html            # 前端页面
├── environment/
│   └── Dockerfile            # Docker 配置
├── requirements.txt          # 依赖清单
├── frozen-requirements.txt   # 锁定版本依赖
└── instruction.md            # 任务说明文档
```

## 快速开始

### 本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker 运行

```bash
# 构建镜像
docker build -f environment/Dockerfile -t ai-ml-demo .

# 运行容器
docker run -p 8000:8000 ai-ml-demo
```

## API 接口

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | / | 前端页面 |
| POST | /train | 触发模型训练 |
| POST | /predict | 执行预测 |
| GET | /metrics | 获取评估指标 |
| GET | /model/status | 检查模型状态 |
| GET | /health | 健康检查 |
