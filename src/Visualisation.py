import matplotlib.pyplot as plt
import matplotlib.animation as animation

# a helper function to make animations, code stolen from:
# https://stackoverflow.com/questions/42386372/increase-the-speed-of-redrawing-contour-plot-in-matplotlib
def ContourPlotAnimator(data
,Times
,Title=None                        
,cmap=plt.get_cmap("viridis")
,cmaplevels=20
,repeat=False
,FrameMin=20):
    
    L=data.shape[0]
    # get the colorbar range
    # don't use the first few frames of data to get the max/min bounds.
    vmax=np.max(data[FrameMin:,:,:])
    vmin=np.min(data[FrameMin:,:,:])
    levels = np.linspace(vmin, vmax,cmaplevels)

    # make the fig
    fig, ax=plt.subplots()
    ax.set_title(Title)

    # make the initial drawing
    p = [ax.contourf(np.transpose(data[0,:,:]), levels, cmap=cmap,vmin=vmin,vmax=vmax)]

    # make the fixed colorbar, and time label
    cbar=fig.colorbar(p[0])
    props = dict(boxstyle='round', facecolor='wheat')
    timelabel = ax.text(0.9,0.9, "", transform=ax.transAxes, ha="right",bbox=props)


    def update(i):
        # remove the old drawing
        for tp in p[0].collections:
            tp.remove()
        p[0] = ax.contourf(np.transpose(data[i,:,:]), levels,cmap=cmap,vmin=vmin,vmax=vmax)  
        label="T="+'{:.2f}'.format(Times[i])   
        timelabel.set_text(label)  
        return p[0].collections

    ani = animation.FuncAnimation(fig, update, frames=L, interval=1, blit=True, repeat=repeat)
    
    return ani

# a function to animatea cross section of the data
def SlicePlotAnimator(data
,Times
,Title=None                        
,cmap=plt.get_cmap("viridis")
,cmaplevels=20
,repeat=False
,FrameMin=20):
    
    L=data.shape[0]
    # get the colorbar range
    # don't use the first few frames of data to get the max/min bounds.
    vmax=np.max(data[FrameMin:,:,:])
    vmin=np.min(data[FrameMin:,:,:])
    levels = np.linspace(vmin, vmax,cmaplevels)

    # make the fig
    fig, ax=plt.subplots()
    ax.set_title(Title)

    # make the initial drawing
    #p = [ax.contourf(np.transpose(data[0,:,:]), levels, cmap=cmap,vmin=vmin,vmax=vmax)]
    #p = [ax.plot(np.transpose(data[0,:,:])]
    line, = ax.plot(x, np.sin(x))

p=ax2.plot(shearmagdata[10,:,:])

    # make the fixed colorbar, and time label
    #cbar=fig.colorbar(p[0])
    #props = dict(boxstyle='round', facecolor='wheat')
    #timelabel = ax.text(0.9,0.9, "", transform=ax.transAxes, ha="right",bbox=props)


    def update(i):
        # remove the old drawing
        for tp in p[0].collections:
            tp.remove()
        p[0] = ax.contourf(np.transpose(data[i,:,:]), levels,cmap=cmap,vmin=vmin,vmax=vmax)  
        label="T="+'{:.2f}'.format(Times[i])   
        timelabel.set_text(label)  
        return p[0].collections

    ani = animation.FuncAnimation(fig, update, frames=L, interval=1, blit=True, repeat=repeat)
    
    return ani

# a few random helper functions
def x(i):
    return (i-((Npts-1)/2))*h

# remember, i is 0 indexed!
def i(x):
    return (x/h)+((Npts-1)/2)
