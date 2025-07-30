# MakeItDrumless

A Python script to automate downloading a YouTube song, removing drums, and generating a backing track for practice or performance.

## Features

- Downloads audio from YouTube
- Splits stems using [SCNet](https://github.com/starrytong/SCNet)
- Removes drums and exports the final mix

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=XXXX"
```

## Requirements

- Python 3.13
- Downloaded SCNet checkpoint files
