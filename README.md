## Initial Setup

### 1. Create Virtual Environment

```bash
python3 -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip freeze > requirements.txt

```

### 3. Run Command

```
python pipeline.py --pdf input/data.pdf --pages 545-550 --test
```
