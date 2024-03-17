import csv
import numpy as np
from predict import estimate_price


def normalize_data(dataset):
    dataset_array = np.array(dataset)

    # Extract the mileages and prices
    mileages = dataset_array[:, 0]
    prices = dataset_array[:, 1]

    # Normalizing the mileages
    normalized_mileages = (mileages - np.mean(mileages)) / np.std(mileages)

    # Return the normalized dataset
    return list(zip(normalized_mileages, prices))


def train_model(dataset, learning_rate, num_iterations):
    theta0 = 0
    theta1 = 0

    m = len(dataset)

    # Perform gradient descent
    for _ in range(num_iterations):
        tmp_theta0 = 0
        tmp_theta1 = 0
        for mileage, price in dataset:
            # Calculating the error
            error = estimate_price(mileage, theta0, theta1) - price

            # Updating the temporary values of theta0 and theta1
            tmp_theta0 += error
            tmp_theta1 += error * mileage

        # Update theta0 and theta1
        theta0 -= (learning_rate * (1 / m) * tmp_theta0)
        theta1 -= (learning_rate * (1 / m) * tmp_theta1)

    return theta0, theta1


def main():
    # Load dataset
    dataset = []
    with open('data.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dataset.append((float(row['km']), float(row['price'])))

    # Normalized dataset for training the model
    normalized_dataset = normalize_data(dataset)

    # Hyperparameters: more learning rate = faster convergence, but risk of
    # overshooting the minimum. Less learning rate = slower convergence,
    # but more accurate
    learning_rate = 0.1
    num_iterations = 1000

    # Train the model
    theta0, theta1 = train_model(
        normalized_dataset,
        learning_rate,
        num_iterations
    )

    # Print the trained parameters
    print("Learning done! Here are the trained parameters:")
    print("Theta0 =", theta0)
    print("Theta1 =", theta1)

    # Save the trained parameters to a file
    with open('parameters.txt', 'w') as file:
        file.write(f"Theta0: {theta0}\nTheta1: {theta1}")


if __name__ == "__main__":
    main()
