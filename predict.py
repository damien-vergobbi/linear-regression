import os
import numpy as np
import matplotlib.pyplot as plt


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def normalize_mileage(mileage, mean, std):
    return (mileage - mean) / std


def main():
    try:
        mileage = float(input("Enter the mileage of the car: "))

        # Basic input validation
        if not isinstance(mileage, (int, float)):
            print("Mileage must be a number!")
            return
        if mileage < 0:
            print("Mileage must be a positive number!")
            return

        # Initialize theta0 and theta1
        theta0 = 0
        theta1 = 0

        # Check if parameters.txt exists
        if os.path.exists('parameters.txt'):
            with open('parameters.txt', 'r') as file:
                lines = file.readlines()
                theta0 = float(lines[0].split(':')[1])
                theta1 = float(lines[1].split(':')[1])

        normalized_mileage = normalize_mileage(mileage, 63060, 31665)
        estimated_price = estimate_price(normalized_mileage, theta0, theta1)

        # Print result
        print(
            "Estimated price for a car with a mileage of {:.0f} km: {:.2f} €"
            .format(mileage, estimated_price)
        )

        # Load dataset
        dataset = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
        mileages = dataset[:, 0]
        prices = dataset[:, 1]

        # Plot the dataset and the linear regression
        plt.scatter(mileages, prices, label='Datas', color='blue', zorder=2)
        plt.plot(
            mileages,
            estimate_price(
                normalize_mileage(mileages, 63060, 31665),
                theta0,
                theta1
            ),
            color='green',
            label='Linear Regression',
            zorder=1
        )

        # Add prediction point
        plt.scatter(
            mileage,
            estimated_price,
            color='red',
            edgecolors='black',
            label='Prediction',
            zorder=3
        )

        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()
    except Exception as e:
        print("An error occurred: ", e)


if __name__ == "__main__":
    main()
