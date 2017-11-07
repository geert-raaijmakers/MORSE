import numpy
import pandas
from matplotlib import pyplot
from scipy.interpolate import griddata
from InputPosteriors import Pobs, find_CI_level

pyplot.rcParams['xtick.direction'] = 'in'
pyplot.rcParams['xtick.minor.visible'] = True
pyplot.rcParams['ytick.direction'] = 'in'
pyplot.rcParams['ytick.minor.visible'] = True
pyplot.rcParams['xtick.major.size'] = 5
pyplot.rcParams['ytick.major.size'] = 5
pyplot.rcParams['ytick.right'] = True
pyplot.rcParams['xtick.top'] = True

pyplot.rcParams['axes.titlesize'] = 15
pyplot.rcParams['axes.labelsize'] = 24
pyplot.rcParams['xtick.labelsize'] = 20
pyplot.rcParams['ytick.labelsize'] = 20
pyplot.rcParams['text.usetex'] = True

def Plot_MRcurves(MRIcurves, num):
    indices = numpy.linspace(0, len(MRIcurves)-1, num, dtype=int)

    fig, ax = pyplot.subplots(1,1, figsize=(7,6))
    for e, i in enumerate(indices):
        M, R, I, rhoc = MRIcurves[i]
        ax.plot(R, M, c='blue', zorder=0)

    ax.set_xlim(5, 16)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel(r'Mass (M$_{\odot}$)')
    pyplot.show()

def Plot_PosteriorInput(PosteriorInput, M, R):
    mi = numpy.linspace(0.2, 3.6, 400)
    ri = numpy.linspace(5, 16, 400)
    mig, rig = numpy.meshgrid(mi, ri)

    pig = Pobs(mig, rig, PosteriorInput)
    fig, ax = pyplot.subplots(1,1, figsize=(7,6))
    for i in range(len(pig)):
        pi = numpy.concatenate(pig[i]).ravel()
        ax.contour(rig, mig, pig[i], linewidth=2.0,
           rstride=1, cstride=1, vmin=numpy.amin(pig[i]), vmax=numpy.amax(pig[i]),
           levels=numpy.array([find_CI_level(pi)[0]]), linestyles='--',
           colors=['red'], extend='max')
    ax.plot(R, M, c='black')
    ax.set_xlim(min(R)-3., 17)
    ax.set_ylim(0., max(M)+.3)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel(r'Mass (M$_{\odot}$)')
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    pyplot.show()

def Plot_Prob(Prob, Parameters):

    df = pandas.DataFrame({'P1':Parameters[:,0],'P2':Parameters[:,1], 'P3':Parameters[:,2], 'Prob':Prob})

    fig, ax = pyplot.subplots(1,2, figsize=(12,5))
    for i, e in enumerate([['P1', 'P2'], ['P2', 'P3']]):
        df2 = df.groupby([e[0], e[1]]).Prob.sum().reset_index()
        df2 = numpy.array(df2)
        values = df2[:,2]
        points = df2[:,0:2]

        values=abs(values)

        X = numpy.log10(points[:,0])
        Y = numpy.log10(points[:,1])
        Z = values

        xi = numpy.linspace(X.min(),X.max(),100)
        yi = numpy.linspace(Y.min(),Y.max(),100)
        zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='cubic')
        xig, yig = numpy.meshgrid(xi, yi)

        surface = ax[i].contour(xig, yig, zi, linewidths=1.5, rstrid=1, cstride=1, vmin=min(Z), vmax=max(Z),
                               levels = find_CI_level(values), colors=('Grey', 'Steelblue'))
        fmt = {}
        strs = [r'2 $\sigma$', r'1 $\sigma$']
        for l, s in zip(surface.levels, strs):
            fmt[l] = s
        ax[i].clabel(surface, inline=1, fontsize=11, fmt=fmt)

        if i==0:
            ### P1 P2 ###
            ax[0].set_xlabel('$\log($P$_{1}$) (dyn cm$^{-2}$)')
            ax[0].set_ylabel('$\log($P$_{2}$) (dyn cm$^{-2}$)')
            ax[0].set_yticks([34.5, 35., 35.5, 36.])
            ax[0].set_xticks([33.5, 34., 34.5, 35.])
            ax[0].text(34.253, 35.115, 'FPS', fontsize=12)
            ax[0].set_xlim(33.5, 35.)
            ax[0].set_ylim(34.2, 36.)

        if i==1:
            ### P2 P3 ###
            ax[1].set_xlabel('$\log($P$_{2}$) (dyn cm$^{-2}$)')
            ax[1].set_ylabel('$\log($P$_{3}$) (dyn cm$^{-2}$)')
            ax[1].set_xticks([34.5, 35., 35.5, 36.])
            ax[1].set_yticks([35., 35.5, 36., 36.5, 37.])
            ax[1].text(35.08, 35.89, 'FPS', fontsize=12)
            ax[1].set_xlim(34.2, 36.)
            ax[1].set_ylim(35., 37.)

    pyplot.tight_layout()
    pyplot.show()







