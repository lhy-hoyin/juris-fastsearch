# Installation Guide

### Create Python virtual environment

```
python3 -m venv .venv
```

### Activate virtual environment

<details>
<summary>On UNIX (macOS/Linux)</summary>

```
source .venv/bin/activate
```
</details>

<details>
<summary>On Windows</summary>

```
.venv\Scripts\activate
```
</details>

### Install dependencies

```
pip install -r requirements.txt
```

### Run application
```
python3 juris.py
```

## Common Errors

### Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0

1. Run 
   ```
   pip install pysqlite3-binary
   ```
2. Open [`juris.py`](../juris.py) and uncomment (not delete!) line 14 by removing `#`
   ```none
   14   """
   15   import chromadb
   16   """
   17   import sys
   18   __import__('pysqlite3')
   19   sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
   20   import chromadb
   21   #"""
   ```
   This will toggle from using the first block (line 15) to using the second block (line 17 - 20)
3. Run `juris.py`

