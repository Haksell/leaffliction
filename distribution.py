from operator import itemgetter
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns


def count_pics(root_dir):
    try:
        storage = dict()
        for subdir in os.listdir(root_dir):
            full_dir = os.path.join(root_dir, subdir)
            if os.path.isdir(full_dir):
                count = sum(
                    file.lower().endswith((".png", ".jpg", ".jpeg"))
                    for file in os.listdir(full_dir)
                )
                if count > 0:
                    storage[subdir] = count
        assert len(storage) != 0, f"No images found in {root_dir}."
        return storage
    except Exception as e:
        print(e)
        sys.exit(1)


def create_chart(storage):
    storage = sorted(storage.items(), key=itemgetter(1), reverse=True)
    labels = [k for k, _ in storage]
    values = [v for _, v in storage]

    colors = sns.color_palette("tab10", n_colors=len(labels))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.pie(values, labels=labels, colors=colors, autopct="%1.1f%%")
    ax1.set_title("Pie chart of class distribution")

    sns.barplot(x=labels, y=values, hue=labels, palette=colors, ax=ax2)
    ax2.set_title("Bar plot of class distribution")
    ax2.set_xlabel("Subdirectories")
    ax2.set_ylabel("Number of Images")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)
    plt.show()


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} directory")
        sys.exit(1)
    create_chart(count_pics(sys.argv[1]))


if __name__ == "__main__":
    main()
