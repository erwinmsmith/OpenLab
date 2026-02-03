<p align="center">
  <img src="assets/OpenLab.png" alt="OpenLab Logo" width="400">
</p>

<h1 align="center">OpenLab</h1>

<p align="center">
  <strong>基于 Claude Code 的邮件触发实验运行器</strong>
</p>

<p align="center">
  <a href="#功能特性">功能特性</a> |
  <a href="#快速开始">快速开始</a> |
  <a href="#使用方法">使用方法</a> |
  <a href="#联系方式">联系方式</a> |
  <a href="README.md">English</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.10+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-1.7+-f7931e.svg" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/platform-linux-lightgrey.svg" alt="Platform">
</p>

---

发送一封包含实验需求的邮件，系统将自动规划、执行并返回结果。

## 功能特性

- **邮件触发实验** - 通过邮件发送实验请求，自动接收结果
- **Claude Code 集成** - AI 驱动的实验规划和代码生成
- **多用户支持** - 按发送者邮箱组织实验运行记录
- **开机自启** - systemd 服务实现持久后台运行
- **完整输出** - 生成规格文件、代码、图表和 PDF 报告
- **Conda 环境** - 预配置的 Python 3.10 环境，包含 ML/DL/统计包

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/erwinmsmith/OpenLab.git
cd OpenLab
```

### 2. 创建 conda 环境

```bash
conda create -n openex python=3.10 -y
conda activate openex
```

### 3. 安装依赖包

```bash
# 核心包
pip install numpy pandas scipy matplotlib seaborn plotly

# 机器学习
pip install scikit-learn xgboost lightgbm catboost

# 深度学习
pip install torch torchvision torchaudio
pip install transformers datasets accelerate timm

# 统计
pip install statsmodels optuna

# 工具
pip install pyyaml
```

### 4. 安装 Claude Code

```bash
# 安装 Node.js (通过 nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# 安装 Claude Code
npm install -g @anthropic-ai/claude-code
```

### 5. 配置环境

```bash
cp .env.example .env
# 编辑 .env 填入你的配置
```

### 6. 启动服务

**方式 A: systemd (推荐用于生产环境)**
```bash
# 编辑 cc-exp-runner.service 匹配你的路径
sudo cp cc-exp-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cc-exp-runner
sudo systemctl start cc-exp-runner
```

**方式 B: screen (用于开发)**
```bash
./screen_start
```

## 配置说明

### .env 文件

复制 `.env.example` 到 `.env` 并填入你的配置：

```bash
# 接收实验请求的邮箱账户
EMAIL_ADDRESS=your-email@example.com

# IMAP 设置 (用于读取邮件)
IMAP_HOST=imap.example.com
IMAP_PORT=993
IMAP_USER=your-email@example.com
IMAP_PASS=your-imap-password

# SMTP 设置 (用于发送回复)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your-email@example.com
SMTP_PASS=your-smtp-password

# 轮询间隔 (秒)
POLL_INTERVAL=60

# Claude Code 路径 (根据你的系统调整)
CLAUDE_PATH=/path/to/claude

# Conda 设置
CONDA_BASE=/path/to/anaconda3
CONDA_ENV=openex
```

### systemd 服务

编辑 `cc-exp-runner.service` 匹配你的系统路径：
- `User`: 你的用户名
- `WorkingDirectory`: OpenLab 目录路径
- `Environment PATH`: 包含你的 node 和 conda 路径
- `ExecStart`: email_listener.py 的路径
- `StandardOutput/StandardError`: 日志文件路径

## 使用方法

### 发送实验请求

向配置的邮箱地址发送邮件：
- **主题**: 实验简要描述
- **正文**: 详细的实验需求

示例：
```
主题: 在鸢尾花数据集上测试 SVM

请在鸢尾花数据集上运行 SVM 分类：
- 使用 3 个不同的随机种子保证可复现性
- 网格搜索超参数 (C, kernel, gamma)
- 生成混淆矩阵和 ROC 曲线
- 使用 bootstrap 置信区间进行统计分析
```

### 查看日志

```bash
# 查看监听器日志
tail -f logs/listener.log

# 查看特定运行的日志
./view_logs runs/<user_email>/<run_id>
```

### 检查服务状态

```bash
sudo systemctl status cc-exp-runner
```

## 项目结构

```
OpenLab/
├── .env.example          # 环境变量模板
├── .gitignore            # Git 忽略规则
├── cc-exp-runner.service # systemd 服务文件
├── config/
│   └── server.yaml.example
├── logs/                 # 监听器日志
├── prompts/
│   └── exp_rules.txt     # 实验生成规则
├── runs/                 # 实验输出 (按用户组织)
│   └── <user_email>/
│       └── <run_id>/
│           ├── spec.yaml
│           ├── run.sh
│           └── artifacts/
│               ├── summary.md
│               ├── report.qmd
│               ├── report.pdf
│               ├── metrics.json
│               ├── figures/
│               ├── tables/
│               └── logs/
├── templates/
│   └── report.qmd        # Quarto 报告模板
├── tools/
│   ├── email_listener.py # 主监听脚本
│   ├── notify_email.py   # 邮件通知
│   ├── feishu_webhook.py # 飞书 webhook
│   └── pack_run.py       # 产物打包
├── screen_start          # screen 启动脚本
├── screen_stop           # screen 停止脚本
├── start_listener        # 启动监听守护进程
├── stop_listener         # 停止监听守护进程
├── view_logs             # 日志查看器
├── test_local.py         # 本地测试脚本
└── exp                   # 手动实验运行器
```

## 支持的实验类型

系统支持多种实验类型：

### 机器学习
- 分类 (SVM, 随机森林, XGBoost, LightGBM, CatBoost)
- 回归
- 聚类
- 超参数调优

### 深度学习
- PyTorch 模型
- Transformers (Hugging Face)
- 自定义神经网络

### 统计分析
- 假设检验
- 回归分析
- Bootstrap 置信区间
- 置换检验

### 可视化
- 训练曲线
- 混淆矩阵
- ROC 曲线
- 决策边界
- 统计图表

## 环境包列表

conda `openex` 环境包含：

| 类别 | 包 |
|------|-----|
| **核心** | numpy, pandas, scipy |
| **机器学习** | scikit-learn 1.7+, xgboost, lightgbm, catboost |
| **深度学习** | torch 2.10+, torchvision, transformers, timm |
| **统计** | statsmodels, optuna |
| **可视化** | matplotlib, seaborn, plotly |
| **工具** | pyyaml, tqdm |

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 联系方式

- **邮箱**: duanzhenke@code-soul.com
- **GitHub**: [@erwinmsmith](https://github.com/erwinmsmith)

## 致谢

- [Claude Code](https://www.anthropic.com/) - AI 驱动的代码生成
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Scikit-learn](https://scikit-learn.org/) - 机器学习库
- [Quarto](https://quarto.org/) - 科学出版系统

## 许可证

MIT License - 详见 [LICENSE](LICENSE)。

---

<p align="center">
  Made with by <a href="mailto:duanzhenke@code-soul.com">Zhenke Duan</a>
</p>
