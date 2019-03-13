import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Plots a picture
def plot2d2(M1, M2, limits, t, v=(None, None)):
    M_min, M_max = v
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im = ax1.imshow(M1, interpolation='bilinear', cmap=cm.Blues_r,
                   origin='lower', extent=limits,
                   vmax=M_max, vmin=M_min)
    fig.colorbar(im, ax=ax1, shrink=0.9)
    preset(ax1, 'x [mcm]', 'y [mcm]', '%.2f' %t + ' ps')
    im = ax2.imshow(M2, interpolation='bilinear', cmap=cm.Blues_r,
                   origin='lower', extent=limits,
                   vmax=M_max, vmin=M_min)
    fig.colorbar(im, ax=ax2, shrink=0.9)
    preset(ax2, 'x [mcm]', 'z [mcm]', '%.2f' %t + ' ps')
#    plt.show()

#Plots a simple picture
def plot2d(M, l, v=(None, None)):
    M_min, M_max = v
    fig, ax = plt.subplots()
    im = ax.imshow(M, interpolation='bilinear', cmap=cm.Blues_r,
                   origin='lower', extent=[l.min(), l.max(), l.min(), l.max()],
                   vmax=M_max, vmin=M_min)
    fig.colorbar(im, ax=ax, shrink=0.9)
#    plt.show()

#Defines picture contents
def plot(E, l, name, fold, time_scale, z_offset=None, mode=None):
    m = int(l.shape[0]/2)
    if mode =='uniform':
        M_min = abs(E).min()
        M_max = abs(E).max()
    else:
        M_min = M_max = None
    i = 0
    while True:
        try:
            if i%1 ==0:
                if z_offset == None:
                    offset = 0
                elif z_offset[i] <= m:
                    offset = m - z_offset[i]
                else:
                    offset = 3*m - z_offset[i]
                z_max = np.argmax(E[i][:,m,m])
                M1 = abs(E[i][z_max])
                M2 = abs(np.concatenate((E[i][offset:,:,m], E[i][0:offset,:,m])))
                l1 = np.round(l[0] * 10**6, 1)
                l2 = np.round(l[-1] * 10**6, 1)
                limits = [l1, l2]
                limits.extend(limits)
                tau = i * time_scale
                plot2d2(M1, M2, limits, tau, (M_min, M_max))
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                filename = fold + '/' + name + f'_{i}.png'
                plt.savefig(filename)
                plt.close(plt.gcf())
            i += 1
        except IndexError:
            break

#Makes animation
def anim(name, fold, duration):
    images = []
    i = 0
    while True:
        try:
            if i%1 ==0:
                filename = fold + '/' + name + f'_{i}.png'
                images.append(imageio.imread(filename))
            i += 1
        except FileNotFoundError:
            break
    imageio.mimsave(fold + '/' + name + '.gif', images, duration=duration)

#Makes labels
def preset(ax, xl, yl, title=None):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if title is not None:
        ax.set_title(title)
