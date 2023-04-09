#!/bin/sh
gunicorn src.views:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000 --timeout 1000