#!/bin/bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 10000 --reload
