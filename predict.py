import os
import numpy as np
import matplotlib.pyplot as plt


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def normalize_mileage(mileage, mean, std):
    return (mileage - mean) / std


def cleanLargeNumber(number):
    if number > 1000000:
        return "{:.0f}M".format(number / 1000000)
    elif number > 1000:
        return "{:.0f}K".format(number / 1000)
    elif number < -1000000:
        return "-{:.0f}M".format(-number / 1000000)
    elif number < -1000:
        return "-{:.0f}K".format(-number / 1000)
    else:
        return "{:.2f}".format(number)


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

        # Open data.csv and get mean and std
        dataset = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
        mileages = dataset[:, 0]
        prices = dataset[:, 1]

        # Statistics
        mean = np.mean(mileages)
        std = np.std(mileages)

        normalized_mileage = normalize_mileage(mileage, mean, std)
        estimated_price = estimate_price(normalized_mileage, theta0, theta1)

        # Print result
        print(
            "Estimated price for a car with a mileage of {:.0f} km: {:.2f} €"
            .format(mileage, estimated_price)
        )

        # Return if no theta0 and theta1
        if theta0 == 0 and theta1 == 0:
            return

        # Precise statistics
        rmse = np.sqrt(
            np.mean(
                (estimate_price(mileages, theta0, theta1) - prices) ** 2
            )
        )
        mae = np.mean(
            np.abs(
                estimate_price(mileages, theta0, theta1) - prices
            )
        )
        r2 = 1 - (
            np.sum(
                (prices - estimate_price(mileages, theta0, theta1)) ** 2
            ) / np.sum((prices - np.mean(prices)) ** 2)
        )

        # Print statistics
        print("RMSE: {:.2f}".format(rmse))  # Root Mean Squared Error
        print("MAE: {:.2f}".format(mae))  # Mean Absolute Error
        print("R2: {:.2f}".format(r2))  # R-squared

        rmse = cleanLargeNumber(rmse)
        mae = cleanLargeNumber(mae)
        r2 = cleanLargeNumber(r2)

        fig, ax = plt.subplots()

        # Plot the dataset and the linear regression
        plt.scatter(mileages, prices, label='Datas', color='blue', zorder=2)
        plt.plot(
            mileages,
            estimate_price(
                normalize_mileage(mileages, mean, std),
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

        # Title with mileage and estimated price
        plt.title(
            "Estimated price of {:.0f} km: {:.2f} €"
            .format(mileage, estimated_price)
        )

        # Add statistics to the plot (bottom left)
        plt.text(
            0.05,
            0.05,
            f"RMSE: {rmse}\nMAE: {mae}\nR2: {r2}",
            transform=ax.transAxes
        )

        plt.legend()
        plt.show()

    except Exception as e:
        print("An error occurred: ", e)


if __name__ == "__main__":
    main()
