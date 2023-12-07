import numpy as np
import matplotlib.pyplot as plt


def generate_synthetic_data(num_samples, image_size):
    data = np.random.randint(0, 2, (num_samples, image_size, image_size))
    labels = np.random.randint(0, 2, num_samples)
    return data, labels


def visualize_data(data, labels, num_images=5):
    for i in range(num_images):
        plt.imshow(data[i], cmap="gray")
        plt.title(f"Label: {labels[i]}")
        plt.show()


if __name__ == "__main__":
    num_samples = 500
    image_size = 10

    data, labels = generate_synthetic_data(num_samples, image_size)
    # visualize_data(data, labels)

    # Save the generated data as data.npy
    np.save("data.npy", data)
    np.save("labels.npy", labels)
