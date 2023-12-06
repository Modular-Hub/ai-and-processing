from OLN import *

if __name__=="__main__":
    classes = 4
    X = np.genfromtxt("Dataset_A03.csv", delimiter=',', skip_header=1, usecols=[0, 1]).T
    Y = np.genfromtxt("Dataset_A03.csv", delimiter=',', skip_header=1, usecols=[2, 3, 4, 5]).T

    net = OLN(2, classes, softMax)
    net.fit(X, Y, lr=1)
    print(net.w)
    print(net.b)
    
    cm = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
    ]

    ax1 = plt.subplot(1, 2, 1)
    y_c = np.argmax(Y, axis=0)
    for i in range(X.shape[1]):
        ax1.plot(X[0, i], X[1, i], '*', c=cm[y_c[i]])
    ax1.axis([-1, 2, -1, 2])
    ax1.grid()
    ax1.set_title("Original Problem")


    ax2 = plt.subplot(1, 2, 2)
    y_c = np.argmax(net.predict(X), axis=0)
    for i in range(X.shape[1]):
        ax2.plot(X[0, i], X[1, i], '*', c=cm[y_c[i]])
    ax2.axis([-1, 2, -1, 2])
    ax2.grid()
    ax2.set_title("Net prediction")

    plt.show()


    """
[[-0.06134116  4.75043293]
 [ 4.61711199 -1.70697038]
 [ 5.38637685 -9.36517564]
 [-9.06565674  6.76654658]]
[[-0.86387677]
 [ 0.07443574]
 [ 0.86480032]
 [-0.39565181]]



    """
