import csv
import numpy as np
import matplotlib.pyplot as plt
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

    # Create plt to show evolution of theta0 and theta1 during training
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    thetas = axs[0]
    affine = axs[1]

    # First plot with the initial values of theta0 and theta1
    thetas.set_title('Evolution of theta0 and theta1 during training')
    thetas.set_xlabel('Iterations')
    thetas.set_ylabel('Theta')
    thetas.grid(True)

    # Show the initial values of theta0 and theta1
    thetas.plot(0, theta0, 'o', color='blue', label='Theta0')
    thetas.plot(0, theta1, 'o', color='red', label='Theta1')
    thetas.legend()

    thetas.set_xlim(0, num_iterations)

    # Second plot with the affine function
    affine.set_title('Affine function')
    affine.set_xlabel('Mileage')
    affine.set_ylabel('Price')
    affine.grid(True)

    # Plot the dataset
    x_values = np.array(dataset)[:, 0]
    y_values = np.array(dataset)[:, 1]
    affine.plot(x_values, y_values, 'o', color='blue')

    # Perform gradient descent
    for i in range(num_iterations):
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

        # Plot the evolution of theta0 and theta1
        thetas.plot(i, theta0, 'o', color='blue')
        thetas.plot(i, theta1, 'o', color='red')

        # Plot the affine function
        x_values = np.linspace(-2, 3, 100)
        y_values = estimate_price(x_values, theta0, theta1)
        affine.plot(
            x_values,
            y_values,
            color=(1, 0, 0) if i == num_iterations - 1 else (0, 1, 0, 0.6)
        )

        plt.draw()

        if i % 50 == 0:
            plt.pause(0.0003)

    plt.show()

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
