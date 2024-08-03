def addline(ax,b,m,*args,**kwargs):
    "Add line with slope m and intercept b"
    xlim = ax.get_xlim()
    ylim = [m*xlim[0]+b,m*xlim[1]+b]
    ax.plot(xlim,ylim,*args,**kwargs)