import numpy as np
from math import cos,sin,pi,isnan
import time
import numpy.matlib as matlib
from qg import modgrid
from qg import moddyn
from qg import modelliptic
import matplotlib.pylab as plt
import pdb
from scipy import interpolate

plt.ioff()
#plt.ion()
# create a directory for plots
import os
plot_dir = "plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

    
def run_qg(h1i=None, h1m=None, Rd=None, beta1x=None, beta1y=None, lon=None, lat=None, tint=None, dtout=None, dt=None, snu1=None):
	
	if snu1 is None: 
		snu1=h1i*0.
	if beta1x is None: 
		beta1x=h1i*0.
	if beta1y is None: 
		beta1y=h1i*0.
	if h1m is None: 
		h1m=h1i*0.
	
	# determine forward or backward
	way=np.sign(tint)
  
	##############
	# Setups
	##############

	grd = modgrid.grid(h1i,Rd,snu1,lon,lat)

	psi1i = grd.g/grd.f0*h1i
	psi1m = grd.g/grd.f0*h1m

	time_abs=0.
	index_time=0    

	nindex_time=int(np.abs(tint)/dtout + 1)
	psi1_sav=np.empty((nindex_time,grd.ny,grd.nx))
	psi1_sav[index_time,:,:]=psi1i

	nstep=int(abs(tint)/dt)
	stepout=int(dtout/dt)

	# Mean fields initializations

	u1m,v1m, = moddyn.psi2uv(psi1m,grd)
	q1m, = modelliptic.psi2pv(psi1m,grd)

	############################
	# Active variable initializations
	############################ 
	psi1 = +psi1i

	q1, = modelliptic.psi2pv(psi1,grd)

	psi1b=+psi1    

	############################
	# Time loop
	############################

	for step in range(nstep): 	
		time_abs=(step+1)*dt
		if (np.mod(step+1,stepout)==0):
			index_time += 1

		############################
		#Initialization of previous fields
		############################    

		psi1guess = 2*psi1 - psi1b
		psi1b = +psi1
		q1_b = +q1
	
		########################
		# Main routines
		########################
    
		# 1/ 
		u1,v1, = moddyn.psi2uv(psi1,grd)

		# 2/
		rq1, = moddyn.qrhs(u1,u1m,v1,v1m,q1_b,q1m,beta1x,beta1y,grd,way)
		rq1diff = moddyn.qrhsdiff(psi1,grd.snu1,grd)
	
		# 3/    
		q1 = q1_b + dt*(rq1+rq1diff)
		#if step%10 == 0:
		#    
		#    #savename = "/data/RESULTS/QG_stochastic/Adv_10jours_PV_QG_V0_%04d.png" %step
		#    fig = plt.figure(figsize=(6, 8))
		#    ax = plt.axes()
		#    cax = ax.pcolormesh(lon, lat, q1 ,shading='gouraud', cmap='seismic', vmin=0.00005, vmax=0.0005)
		#    ax.set(title='PV: t = ' + str(round(step,4)))
		#    cbar = fig.colorbar(cax)
		#    
		#    fig.savefig(plot_dir + '/PV_dt300sQG_v0_{:0>4d}.png'.format(step),bbox_inches='tight')
		#    plt.close()
		
		# 4/
		psi1 = modelliptic.pv2psi(q1,psi1guess,grd)

		############################
		#Saving outputs
		############################

		if (np.mod(step + 1, stepout) == 0):
			psi1_sav[index_time, :, :] = psi1
			#fig, (ax1) = plt.subplots(figsize=(6, 8), ncols=1);plt.pcolormesh(q1, cmap='seismic', vmin=0.00005, vmax=0.0005); plt.colorbar();  
      

	h1=grd.f0/grd.g*psi1_sav

	return h1
