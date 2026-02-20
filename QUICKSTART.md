# Quick Start - Mini Agentic AI

Get the system running in 3 steps.

## 1️⃣ Get API Key (2 min)

1. Go to: https://aistudio.google.com/app/apikey
2. Click **Create API Key**
3. Copy your key

## 2️⃣ Setup (1 min)

```bash
cd mini_agent
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
export GEMINI_API_KEY="your-key"
```

## 3️⃣ Run Server (Start)

```bash
python server.py
```

Open browser: **http://localhost:8000**

## First Task

In the dashboard, enter a goal:
```
Create a Python script that counts files in the current directory
```

Click **START RUN** and watch execution in real-time.

## What Happens

1. Planner breaks goal into steps
2. Coder writes the script
3. Files appear in `workspace/`
4. All events stream live to dashboard

---

**Need more details?** See [README.md](README.md)
