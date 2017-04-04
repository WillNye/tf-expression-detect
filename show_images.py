import numpy as np
import matplotlib.pyplot as plt

from util import get_data


label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def main():
    X, Y = get_data(balance_ones=False)

    while True:
        for label in range(7):
            x, y = X[Y == label], Y[Y == label]
            j = np.random.choice(len(y))
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(label_map[y[j]])
            plt.show()

        continue_viewing = input('Continue? Yes or No\n')
        continue_viewing = continue_viewing[0].upper()
        if continue_viewing != 'Y':
            return


if __name__ == '__main__':
    main()


