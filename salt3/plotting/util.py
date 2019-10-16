import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.gridspec as gridspec

def plot(plot_type,x,y=None,yerr=None,xerr=None,x_lab='',y_lab='',fontsize=18,figsize=(12,12),**kwargs):
	fig=plt.figure(figsize=figsize)
	ax=fig.gca()
	if plot_type=='scatter':
		ax.scatter(x,y,**kwargs)
	elif plot_type=='plot':
		ax.plot(x,y,**kwargs)
	elif plot_type=='errorbar':
		ax.errorbar(x,y,xerr=xerr,yerr=yerr,**kwargs)
	elif plot_type=='hist':
		ax.hist(x,**kwargs)
	else:
		raise RuntimeError('What plot are you trying to do.')
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax.set_xlabel(x_lab,fontsize=fontsize)
	ax.set_ylabel(y_lab,fontsize=fontsize)
	return(ax)

def split_plot(ax,plot_type,x,y=None,yerr=None,xerr=None,x_lab='',y_lab='',xticks=False,fontsize=18,**kwargs):
	ax_divider = make_axes_locatable(ax)
	ax_ml = ax_divider.append_axes("bottom", size="50%", pad=.2)
	ticks=[]
	for tick in ax_ml.xaxis.get_major_ticks():
		ticks.append('')
		tick.label.set_fontsize(14)
	if not xticks:
		ax.set_xticklabels(ticks)
	for tick in ax_ml.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	if plot_type=='scatter':
		ax_ml.scatter(x,y,**kwargs)
	elif plot_type=='plot':
		ax_ml.plot(x,y,**kwargs)
	elif plot_type=='errorbar':
		ax_ml.errorbar(x,y,xerr=xerr,yerr=yerr,**kwargs)
	elif plot_type=='hist':
		ax_ml.hist(x,**kwargs)
	else:
		raise RuntimeError('What plot are you trying to do.')
	for tick in ax_ml.xaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	for tick in ax_ml.yaxis.get_major_ticks():
		tick.label.set_fontsize(14)
	ax_ml.set_xlabel(x_lab,fontsize=fontsize)
	ax_ml.set_ylabel(y_lab,fontsize=fontsize)
	return(ax,ax_ml)

def grid_plot(grid_x,grid_y,figsize=(12,12)):
	fig=plt.figure(figsize=figsize)
	gs = gridspec.GridSpec(2, 2)
	return(fig,gs)




