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

## Video Walkthrough - Codespace

<video src="Codespace_setup.mp4" controls></video>

https://github.com/user-attachments/assets/f79fc252-400e-48f6-b432-6e75bf37af03

<details open>
<summary>Skip to sections</summary>
00:00 - Create Python virtual environment<br>
00:13 - Activate virtual environment<br>
00:18 - Install dependencies<br>
04:20 - sqlite3 error (Error + how to fix)<br>
05:15 - Run Application<br>
</details>

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
3. Save and run `juris.py`

