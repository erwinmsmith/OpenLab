# OpenLab

Email-triggered experiment runner powered by Claude Code. Send an email with your experiment request, and the system automatically plans, executes, and returns results.

## Features

- **Email-triggered experiments**: Send experiment requests via email, receive results automatically
- **Claude Code integration**: AI-powered experiment planning and code generation
- **Multi-user support**: Organizes runs by sender email address
- **Auto-start on boot**: systemd service for persistent background operation
- **Comprehensive outputs**: Generates spec, code, figures, tables, and PDF reports
- **Conda environment**: Pre-configured Python 3.10 environment with ML/DL/statistics packages

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/erwinmsmith/OpenLab.git
cd OpenLab
```

### 2. Create conda environment

```bash
conda create -n openex python=3.10 -y
conda activate openex
```

### 3. Install required packages

```bash
# Core packages
pip install numpy pandas scipy matplotlib seaborn plotly

# Machine Learning
pip install scikit-learn xgboost lightgbm catboost

# Deep Learning
pip install torch torchvision torchaudio
pip install transformers datasets accelerate timm

# Statistics
pip install statsmodels optuna

# Utilities
pip install pyyaml
```

### 4. Install Claude Code

```bash
# Install Node.js (via nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 20
nvm use 20

# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

### 5. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 6. Start the service

**Option A: systemd (recommended for production)**
```bash
# Edit cc-exp-runner.service to match your paths
sudo cp cc-exp-runner.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cc-exp-runner
sudo systemctl start cc-exp-runner
```

**Option B: screen (for development)**
```bash
./screen_start
```

## Configuration

### .env file

Copy `.env.example` to `.env` and fill in your values:

```bash
# Email account for receiving experiment requests
EMAIL_ADDRESS=your-email@example.com

# IMAP settings (for reading incoming emails)
IMAP_HOST=imap.example.com
IMAP_PORT=993
IMAP_USER=your-email@example.com
IMAP_PASS=your-imap-password

# SMTP settings (for sending replies)
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your-email@example.com
SMTP_PASS=your-smtp-password

# Poll interval (seconds)
POLL_INTERVAL=60

# Claude Code path (adjust for your system)
CLAUDE_PATH=/path/to/claude

# Conda settings
CONDA_BASE=/path/to/anaconda3
CONDA_ENV=openex
```

### systemd service

Edit `cc-exp-runner.service` to match your system paths:
- `User`: your username
- `WorkingDirectory`: path to OpenLab directory
- `Environment PATH`: include your node and conda paths
- `ExecStart`: path to email_listener.py
- `StandardOutput/StandardError`: path to log file

## Usage

### Send experiment request

Send an email to your configured email address with:
- **Subject**: Brief description of experiment
- **Body**: Detailed experiment request

Example:
```
Subject: Test SVM on iris dataset

Please run SVM classification on the iris dataset with:
- 3 different seeds for reproducibility
- Grid search for hyperparameters (C, kernel, gamma)
- Generate confusion matrix and ROC curves
- Statistical analysis with bootstrap confidence intervals
```

### View logs

```bash
# View listener log
tail -f logs/listener.log

# View specific run logs
./view_logs runs/<user_email>/<run_id>
```

### Check service status

```bash
sudo systemctl status cc-exp-runner
```

## Project Structure

```
OpenLab/
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
├── cc-exp-runner.service # systemd service file
├── config/
│   └── server.yaml.example
├── logs/                 # Listener logs
├── prompts/
│   └── exp_rules.txt     # Experiment generation rules
├── runs/                 # Experiment outputs (per user)
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
│   └── report.qmd        # Quarto report template
├── tools/
│   ├── email_listener.py # Main listener script
│   ├── notify_email.py   # Email notification
│   ├── feishu_webhook.py # Feishu webhook
│   └── pack_run.py       # Artifact packaging
├── screen_start          # Start with screen
├── screen_stop           # Stop screen session
├── start_listener        # Start listener daemon
├── stop_listener         # Stop listener daemon
├── view_logs             # Log viewer
├── test_local.py         # Local testing script
└── exp                   # Manual experiment runner
```

## Supported Experiment Types

The system supports various experiment types:

### Machine Learning
- Classification (SVM, Random Forest, XGBoost, LightGBM, CatBoost)
- Regression
- Clustering
- Hyperparameter tuning

### Deep Learning
- PyTorch models
- Transformers (Hugging Face)
- Custom neural networks

### Statistics
- Hypothesis testing
- Regression analysis
- Bootstrap confidence intervals
- Permutation tests

### Visualization
- Training curves
- Confusion matrices
- ROC curves
- Decision boundaries
- Statistical plots

## Environment Packages

The conda `openex` environment includes:

| Category | Packages |
|----------|----------|
| **Core** | numpy, pandas, scipy |
| **ML** | scikit-learn 1.7+, xgboost, lightgbm, catboost |
| **DL** | torch 2.10+, torchvision, transformers, timm |
| **Statistics** | statsmodels, optuna |
| **Visualization** | matplotlib, seaborn, plotly |
| **Utilities** | pyyaml, tqdm |

## License

MIT License
