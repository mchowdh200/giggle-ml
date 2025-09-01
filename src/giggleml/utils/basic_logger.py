class BasicLogger:
    def __init__(self):
        self.contents = list()

    def print(self, *message):
        self.contents.append(" ".join(map(str, message)))

    def save(self, path):
        with open(path, "w") as f:
            f.write(str(self))

    def __str__(self):
        return "\n".join(self.contents)
