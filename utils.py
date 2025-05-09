import json

def load_data(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f]
