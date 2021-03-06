import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Plots a picture
def plot2d2(M1, M2, limits1, limits2, t, v=(None, None)):
    M_min, M_max = v
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im = ax1.imshow(M1, interpolation='lanczos', cmap=cm.RdBu,
                   origin='lower', extent=limits1,
                   vmax=M_max, vmin=M_min)
    cbar = fig.colorbar(im, ax=ax1, shrink=0.9)
    cbar.set_label('Intensity [mW/m^2]')
    preset(ax1, 'x [mcm]', 'y [mcm]', '%.2f' %t + ' fs')
    im = ax2.imshow(M2, interpolation='lanczos', cmap=cm.RdBu,
                   origin='lower', extent=limits2,
                   vmax=M_max, vmin=M_min)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.9)
    cbar.set_label('Intensity [mW/m^2]')
    preset(ax2, 'x [mcm]', 'z [mcm]', '%.2f' %t + ' fs')
    fig.suptitle('Laguerre-Gaussian type of pulse\n\n Parameters: wavelength - 404 nm; waist - 4 mcm; time duration - 0.534 ps; full energy - 10 mJ; order - l=1, q=1', fontsize=14)
    return fig
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
def plot(E, l, name, fold, time_scale, z_range, delimiter, z_offset=None, mode=None):
    m = int(l.shape[0]/2)
    figures = []
    E = E * 10**5
    if mode =='uniform':
        M_min = abs(E).min()
        M_max = abs(E).max()
    else:
        M_min = M_max = None
    i = 0
    while i < 30:
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
                l1 = np.round(l[0], 1)
                l2 = np.round(l[-1], 1)
                z1 = np.round(z_range[0], 1)
                z2 = np.round(z_range[-1]/10, 1)
                limits1 = [l1, l2, l1, l2]
                limits2 = [l1, l2, z1, z2]
                tau = i * time_scale
                fig = plot2d2(M1, M2, limits1, limits2, tau, (M_min, M_max))
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
            i += 1
            figures.append(fig)
        except IndexError:
            break
    return figures

#Makes animation
def anim(name, fold, duration, delimiter):
    images = []
    i = 0
    while True:
        try:
            if i%1 ==0:
                filename = fold + delimiter + name + f'_{i}.png'
                images.append(imageio.imread(filename))
            i += 1
        except FileNotFoundError:
            break
    imageio.mimsave(fold + delimiter + name + '.gif', images, duration=duration)

#Makes labels
def preset(ax, xl, yl, title=None):
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    if title is not None:
        ax.set_title(title)
