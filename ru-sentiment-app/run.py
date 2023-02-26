"""Script for running service."""
import uvicorn

from src import app, configurations

if __name__ == "__main__":
    uvicorn.run(app, host=configurations.HOST, port=configurations.PORT)
