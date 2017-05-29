import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import statsmodels.api as sm


columns = [
    'Day/ Treatment',
    '0,0', '0,3', '0,6', '0,9',
    '0.25,0', '0.25,3', '0.25,6', '0.25,9',
    '0.5,0', '0.5,3', '0.5,6', '0.5,9',
    '0.75,0', '0.75,3', '0.75,6', '0.75,9'
]


def actual_number(val):
    return val != ' ' and \
        val != ',' and \
        val != ', '


def convert_to_df():
    with open('ScallionData.csv') as csvfile:
        plant_reader = csv.reader(csvfile)
        scallion_data = [[[height
                           for height in treatment.split(',')
                           if actual_number(height)]
                          for treatment in day
                          if actual_number(day)]
                         for day in plant_reader]
        return pd.DataFrame(scallion_data[1:], columns=columns)


def convert_to_np(scallion_data, treatment):
    scallion_X = np.array([2 * int(x[0])
                           for x in scallion_data['Day/ Treatment']
                           for _ in range(5)])

    scallion_y = np.array([[float(y)
                            for y in row]
                           for row in scallion_data[treatment]])
    scallion_y = np.reshape(scallion_y, -1)

    return scallion_X, scallion_y


def linear_model(X, y):
    """Builds a linear model and returns a statsmodels object"""
    results = sm.OLS(y, sm.add_constant(X)).fit()
    return results


def resids(X, y, model):
    """Returns a numpy array contatining residuals"""
    return np.array([y_coord - (X[index] * model.params[1] + model.params[0])
                     for index, y_coord in enumerate(y)])


def plot_data(scallion_X, scallion_y, treatment):
    fig = plt.gcf()
    fig.canvas.set_window_title('Plant Data')

    results = linear_model(scallion_X, scallion_y)
    X_plot = np.linspace(0, 12, 100)
    plt.plot(X_plot, X_plot * results.params[1] + results.params[0])

    plt.xlabel('Time (days)')
    plt.ylabel('Height (cm)')
    plt.title("{} Scallion Growth Rate".format(treatment))
    plt.legend()

    plt.scatter(scallion_X, scallion_y, label='height', color='k', s=25, marker='o')
    plt.show()


def plot_resids(X, y, treatment):
    fig = plt.gcf()
    fig.canvas.set_window_title('Plant Residuals')

    y = resids(X, y, linear_model(X, y))
    plt.xlabel('Time (days)')
    plt.ylabel('Residuals (cm)')
    plt.title("{} Scallion Growth Rate Residuals".format(treatment))
    plt.legend()

    plt.scatter(X, y, label='height', color='k', s=25, marker='o')
    plt.show()


if __name__ == '__main__':
    scallion_data = convert_to_df()
    for treatment in columns[1:]:
        X, y = convert_to_np(scallion_data, treatment)
        plot_data(X, y, treatment)
        plot_resids(X, y, treatment)
