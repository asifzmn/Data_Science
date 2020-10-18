import matplotlib.pyplot as plt
import numpy as np


def visualize(x, y, z, C, T, A):
    fig = plt.figure()

    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, c=C, cmap='Purples')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(T)

    if (A != -1): ax.view_init(azim=A)

    plt.show()
    # plt.savefig(T+""+str(A))


def Visualization(subjects):
    while True:
        print()
        print("Enter Option for FMRI reading Visualization")
        print()
        print("1   Select trial with person number")
        print("0   Return")

        option = int(input())

        if (option == 1):

            print("Enter Person Number (1-9)")
            person = int(input()) - 1
            print("Enter Trial Number (1-360)")
            trial = int(input()) - 1

            subject = subjects[person]
            meta = subject.meta
            info = subject.info
            data = subject.dataset
            part = []
            for x in range(meta.dimx):
                for y in range(meta.dimy):
                    for z in range(meta.dimz):
                        if (meta.coordToCol[x][y][z] != 0):
                            # part.append(meta.coordToCol[x][y][z])
                            part.append([x, y, z, data[trial][meta.coordToCol[x][y][z] - 1]])

            part = np.array(part)
            part = part[::15]

            visualize(part[:, 0], part[:, 1], part[:, 2], part[:, 3], "Brain response to word '" + info[trial][2] + "'",
                      -1)

        if (option == 0): return
