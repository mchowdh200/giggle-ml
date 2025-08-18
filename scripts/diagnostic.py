import os
import pickle

from matplotlib import pyplot as plt


def main():
    for name in os.listdir("data/roadep_combo"):
        a = pickle.load(
            open(
                f"data/roadep_combo/{name}",
                "rb",
            )
        )

        items = list(a.items())
        items.sort(key=lambda x: x[1])
        [print(x) for x in items]

        floats = [x[1] for x in items]
        plt.hist(floats, bins=20)
        small = sum([x[1] < 0.05 for x in items]) / 1905 * 100
        top = max([x[1] for x in items])
        plt.title(f"{small}% | {top}")
        plt.show()

        plt.plot(floats)
        plt.title(f"{small}% | {top}")
        plt.show()


if __name__ == "__main__":
    main()
