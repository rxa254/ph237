
# coding: utf-8

# # Example Bayesian Monte Carlo
# 
# This notebook follows Kevin's example of doing a Bayesian Monte Carlo analysis of a decaying sinusoidal function with noise, applied to the Mercury precession problem.

# In[1]:


from __future__ import division
import numpy as np
import emcee
from numpy.random import randn, rand
import matplotlib.gridspec as gridspec
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import interpolate
from scipy import optimize
import astropy.constants as const

from timeit import default_timer as timer  # this is for timing the ODE solvers

get_ipython().magic(u'matplotlib inline')

# uncomment if you have a Mac with Retina display
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")

mpl.rcParams.update({'text.usetex': False,
                     'lines.linewidth': 2.5,
                     'font.size': 18,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'axes.labelsize': 'large',
                     'axes.grid': True,
                     'grid.alpha': 0.73,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 13,
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.1,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'savefig.dpi': 100,
                     'figure.figsize': (9,6),
                     'pdf.compression': 9})


# ## Set up the model
# 
# The model is provided by the JPL ephemeris ODE solver for the precession of mercury. We assume this solver gives us a precession value that we can compare to measurements.
# 
# The parameters are the Sun's quadrupole moment (magnitude and direction)

# In[2]:


# some constants
a = 0.38709893 * const.au.value;    # average radius
e = 0.20563069     # eccentricity
m = 0.33011 * 10**24 #/ const.M_sun.value # mass of mercury (units of solar mass)
M = const.M_sun.value                                  # mass of sun


asecprad = 180/np.pi * 3600
dayspy   = 365.25         # Julian years
secspd   = 86400          # seconds per day
secspy   = dayspy * secspd

G = const.G.value #/ (const.au.value**3) * const.M_sun.value * secspy**2  # Newton [AU**3 / Msun / year**2]
c = const.c.value #/  const.au.value  * secspy   # speed of light [AU / year]
period = 2*np.pi * np.sqrt(a**3 / (G*M))
L = m * np.sqrt(G * M * a * (1-e**2)) # mercury's angular momentum
l = L / m


# In[3]:


n_Earth_years = 1000
n_steps = n_Earth_years * (secspy / period) * 300
t = np.linspace(0, n_Earth_years*secspy, int(n_steps)) # time in units of seconds

tol = 1e-11
#       r   , rdot, phi
r0 = a * (1-e)
y0 = [r0, 0.0, 0.0]
#print(t)


# In[4]:


def rad_time_GR(y, t, J2, Jtheta):
    r, rdot, phi  = y
    dydt = [rdot, -G*M/r**2 + l**2/r**3 - J2*G*(l**2)*M*np.cos(Jtheta)/(2 * c**2 * r**4), l/r**2] 
    # the equation for d(rdot)/dt includes the GR term
    return dydt


# In[5]:


def signalFunction(J2, Jtheta):
    sol_GR = integrate.odeint(rad_time_GR, y0, t, args=(J2, Jtheta), atol=tol, rtol = tol, printmessg=True, hmax=period/4)
    fr_GR   = interpolate.splrep(t, sol_GR[:,0]) #interpolate radius with time
    fphi_GR = interpolate.splrep(t, sol_GR[:,2]) #interpolate phi with time

    time_periapsis = [] # record time of periapsis
    delta_per      = [] # record shift in periapsis (in each orbit)
    per_rate       = [] # record the average rate of shift in periapsis 

    N = n_Earth_years * secspy / period # how many periods to monitor

    for ii in range(int(N-202), int(N-2)):
        # find time at which r is the same as periapsis again
        sol = optimize.root(lambda t : interpolate.splev(t, fr_GR, der=0) - y0[0], ii*period)
        time_periapsis = np.append(time_periapsis, sol.x) #record time of periapsis
        # calculate the change in phi, this gives a cumulative change
        delta_per      = np.append(delta_per,
                                   interpolate.splev(sol.x, fphi_GR, der=0) - 2*ii*np.pi)
        # rate at which phi changes (radians / second)
        per_rate       = np.append(per_rate, delta_per[-1]/(time_periapsis[-1])) 
    
    precession = asecprad * per_rate * (100*secspy)
    return np.mean(precession) # output of ODE, radians of precession

#print(signalFunction((6,0)))
# returns the log likelihood of the observed data given the model,
# assuming gaussian normal errors
def lnlike(theta, data):
    J2, Jtheta = theta
    model = signalFunction(J2, Jtheta)
    return -0.5 * np.sum((data - model)**2)

# Sean and Aaron's priors
Jcutoff0 = 10
def lnprior(theta, Jcutoff=Jcutoff0):
    J2, Jtheta = theta
    '''if not 0 < J2 < Jcutoff:
        return -np.inf
    else:
        return -J2'''
    return 1

def lnpost(theta, tt, data):
    return lnlike(theta, data) + lnprior(theta)
#print(lnpost([6,0], 0, 43))


# In[6]:


# Measured precession of Mercury in arcsec/century
data = 43.0
tt = 0


# ## Setup the Monte Carlo Sampling

# In[7]:


ndim = 2 # number of parameters to estimate
nwalkers = 160 # number of walkers
nsteps = 1500 # number of steps each walker will take
nburn = int(nsteps/10) # number of steps to "burn in"
nthreads = 16 # number of parallel threads to use


# In[8]:


# Choose initial conditions
# We need to know the scaling of the ODE integrator
theta0 = np.array(
    [[rand()*10, rand()*2*np.pi]
     for ii in range(nwalkers)])


# In[ ]:


# Run the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, args=(tt, data), threads=nthreads);
sampler.run_mcmc(theta0, nsteps);


# ## Plot the Results

# In[87]:


fig = plt.figure(figsize=(11, 18));
gs = gridspec.GridSpec(3, 1, hspace=0.05);
J2_ax = fig.add_subplot(gs[0]);
Jtheta_ax = fig.add_subplot(gs[1], sharex=J2_ax);
for ii in range(0, nwalkers):
    J2_ax.plot(sampler.chain[ii, :, 0]);
    Jtheta_ax.plot(np.mod(sampler.chain[ii, :, 1], 2*np.pi));
plt.setp(J2_ax.get_xticklabels(), visible=False);
plt.setp(Jtheta_ax.get_xticklabels(), visible=False);
J2_ax.set_ylabel(r'$J_2$');
Jtheta_ax.set_ylabel(r'$J_{\theta}$');
Jtheta_ax.set_xlabel('step');
Jtheta_ax.set_xlim(0, nsteps);
for ax in [J2_ax, Jtheta_ax]:
    ax.grid('on', which='both', alpha=0.3);
    ax.grid(which='minor', alpha=0.2);

plt.savefig("Burning.pdf", bbox_inches='tight')


# In[88]:


# Only take the samples after burning in the sampler to remove effects of initial conditions
samples = sampler.chain[:, nburn:, :].reshape((-1, ndim))


# In[89]:


# Make a corner plot
fig = corner.corner(samples, labels=[r'$J_2$', r'$J_{\theta}$']);
fig.set_size_inches((12, 12));
plt.savefig("Burning.pdf", bbox_inches='tight')

