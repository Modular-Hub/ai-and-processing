import numpy as np
import matplotlib.pyplot as plt

def draw_2d(model, i, last = False) -> None:
    w1, w2, b = model.w[0], model.w[1], model.b
    li, ls = -2, 2

    if ( last ):
        plt.plot(
            [li, ls],
            [(1/w2)*(-w1*(li)-b), (1/w2)*(-w1*(ls)-b)],
            '--k', label='last'
        )

    else:
        plt.plot(
            [li, ls],
            [(1/w2)*(-w1*(li)-b), (1/w2)*(-w1*(ls)-b)]
        )

def get_normalized(N):
    Y = np.zeros(N)
    weight_lower, weight_upper = 35, 120
    height_lower, height_upper = 1.2, 2.1

    ## Random Inputs
    X = np.array([
        (weight_lower + (weight_upper - weight_lower) * np.random.rand(N)),
        (height_lower + (height_upper - height_lower) * np.random.rand(N))
    ])

    ## Out
    for i in range(N):
        if X[0, i] / X[1, i]**2 < 25:
            Y[i] = 0
        else:
            Y[i] = 1

    ## Normalization
    nmin, nmax = min(X[0, :]), max(X[0, :])

    for i in range(N):
        X[0, i] = (( X[0, i] - nmin) / (nmax - nmin))

    return X, Y

class Perceptron:
    def __init__(self, n_input, learning_rate) -> None:
        self.w = -1 + 2 * np.random.rand(n_input)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate

    def predict(self, X) -> bool:
        _, p = X.shape
        y_est = np.zeros(p)

        for i in range(p):
            y_est[i] = np.dot(self.w, X[:, i]) + self.b
            
            if ( y_est[i] >= 0 ):
                y_est[i] = 1
            else:
                y_est[i] = 0

        return y_est
    
    def fit(self, X, Y, epoch = 10) -> None:
        it = 1
        _, p = X.shape
        for _ in range(epoch):
            if ( _ == epoch-1 ):
                draw_2d(neuron, it, True)
            else:
                draw_2d(neuron, it)

            for i in range(p):
                y_est = self.predict(X[:, i].reshape(-1, 1))
                self.w += self.eta * (Y[i] - y_est) * X[:, i]
                self.b += self.eta * (Y[i] - y_est)

                it+=1

if __name__=='__main__':
    neuron = Perceptron(2, 0.2)
    problem = "bmi"   #[ "gates" "bmi" ]

    ## Logic gates
    if problem == "gates":
        X = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
        ])
        Y = np.array([1, 0, 0, 1])
    ## BMI
    else:
        N = 100
        X, Y = get_normalized(N)
        

    print('W:\t', neuron.w )
    print('P:\t', neuron.predict(X))
    neuron.fit(X, Y)
    print('W:\t', neuron.w )
    print('P:\t', neuron.predict(X))

    ##  Point draws
    _, p = X.shape
    for i in range(p):
        if(not Y[i]):
            plt.plot(X[0, i], X[1, i], 'or')
        else:
            plt.plot(X[0, i], X[1, i], 'ob')


    plt.title('BMI')
    plt.grid('on')
    plt.xlim([-1, 3])
    plt.ylim([0, 3])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$y_1$')
    plt.legend(loc='upper left')
    plt.show()


    ## Second date test for BMI
    if problem == "bmi":
        plt.figure()
        X, Y = get_normalized(N)

        for i in range(N):
            Y[i] = neuron.predict(X[:, i].reshape(-1, 1))

        ##  Point draws
        _, p = X.shape
        for i in range(p):
            if(not Y[i]):
                plt.plot(X[0, i], X[1, i], 'or')
            else:
                plt.plot(X[0, i], X[1, i], 'ob')

        draw_2d(neuron, 0, True)

        plt.title('Second BMI')
        plt.grid('on')
        plt.xlim([-1, 3])
        plt.ylim([0, 3])
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$y_1$')
        plt.legend(loc='upper left')
        plt.show()

