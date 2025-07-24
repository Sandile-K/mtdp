#!/bin/bash
systemctl stop ollama.service
ssh -J ubuntu@196.24.241.109 ubuntu@10.0.0.6 -L 11434:localhost:11434
