from math import exp, sqrt, log, pi
from scipy.optimize import newton, curve_fit
import pandas as pd
from scipy import stats

'''Standard normal distribution function and density'''
Phi = stats.norm.cdf
phi = stats.norm.pdf

'''
Notation:
  v - option price
  k - strike
  x - underlying asset price
  tau - time to expiration
  r - risk-free interest rate
'''

'''d1 and d2 values from Black-Scholes formula'''
def bs_d1(k, x, tau, sigma, r=0):
    return 1/(sigma*sqrt(tau)) * (log(x/k) + (r+sigma**2/2)*tau)

def bs_d2(k, x, tau, sigma, r=0):
    return bs_d1(k,x,tau,sigma,r) - sigma*sqrt(tau)

'''Black-Scholes formula for call and put options'''
def call_price(k, x, tau, sigma, r=0):
    return Phi(bs_d1(k,x,tau,sigma,r))*x  - Phi(bs_d2(k,x,tau,sigma,r))*k*exp(-r*tau)

def put_price(k, x, tau, sigma, r=0):
    return Phi(-bs_d2(k,x,tau,sigma,r))*k*exp(-r*tau) - Phi(-bs_d1(k,x,tau,sigma,r))*x

'''Greeks for Black-Scholes model
  delta = dv / dx
  theta = - dv / dtau
  vega = dv / dsigma
  rho = dv / dr
  gamma = d^2v/ dx^2
  vomma = d^2v / dsigma^2

  Note: vomma is used only in Newton's method
    for computing implied volatility
'''
def call_delta(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = call_iv(v,k,x,tau,r)
    return Phi(bs_d1(k,x,tau,sigma,r))

def put_delta(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = put_iv(v,k,x,tau,r)
    return -Phi(-bs_d1(k,x,tau,sigma,r))

def call_theta(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = call_iv(v,k,x,tau,r)
    return -x*phi(bs_d1(k,x,tau,sigma,r))*sigma/(2*sqrt(tau)) - r*k*exp(-r*tau)*Phi(bs_d2(k,x,tau,sigma,r))

def put_theta(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = put_iv(v,k,x,tau,r)
    return -x*phi(bs_d1(k,x,tau,sigma,r))*sigma/(2*sqrt(tau)) + r*k*exp(-r*tau)*Phi(-bs_d2(k,x,tau,sigma,r))

def call_vega(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = call_iv(v,k,x,tau,r)
    return x*phi(bs_d1(k,x,tau,sigma,r))*sqrt(tau)

def put_vega(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = put_iv(v,k,x,tau,r)
    return x*phi(bs_d1(k,x,tau,sigma,r)) * sqrt(tau)

def call_rho(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = call_iv(v,k,x,tau,r)
    return k*tau*exp(-r*tau)*Phi(bs_d2(k,x,tau,sigma,r))

def put_rho(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = put_iv(v,k,x,tau,r)
    return -k*tau*exp(-r*tau)*Phi(-bs_d2(k,x,tau,sigma,r))

def call_gamma(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = call_iv(v,k,x,tau,r)
    return phi(bs_d1(k,x,tau,sigma,r))/(x*sigma*sqrt(tau))

def put_gamma(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = put_iv(v,k,x,tau,r)
    return phi(bs_d1(k,x,tau,sigma,r))/(x*sigma*sqrt(tau))

def call_vomma(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = call_iv(p,k,x,tau,r)
    return call_vega(k,x,tau,sigma,r)*bs_d1(k,x,tau,sigma,r)*bs_d2(k,x,tau,sigma,r)/sigma

def put_vomma(k, x, tau, sigma=None, r=0, v=None):
    if sigma is None:
        sigma = put_iv(p,k,x,tau,r)
    return put_vega(k,x,tau,sigma,r)*bs_d1(k,x,tau,sigma,r)*bs_d2(k,x,tau,sigma,r)/sigma

'''
Functions for computing implied volatility (IV).

IV for calls is computed in call_iv by solving the Black-Scholes formula for sigma.
The Newton-Halley method is used (a modification of Newton's method using the second
derivative, with cubic convergence rate). This requires two derivatives of v with
respect to sigma: vega and vomma.

Put calculations are reduced to calls via put-call parity.

call_iv_estimate provides a good initial approximation for sigma using the
Corrado & Miller (1996) method.
'''

def call_iv_estimate(v, k, x, tau, r=0.0525):
    z = exp(-r*tau)*k
    a = max((v-(x-z)/2)**2 - (x-z)**2/pi, 0)
    return sqrt(2*pi/tau)/(x+z)*(v-(x-z)/2 + sqrt(a))

def call_iv(v, k, x, tau, r=0.0525):
    if v <= 0:
        return float('NaN')
    try:
        return newton(lambda sigma: call_price(k,x,tau,sigma,r) - v,
                      call_iv_estimate(v,k,x,tau,r),
                      fprime = lambda sigma: call_vega(k,x,tau,sigma,r),
                      fprime2 = lambda sigma: call_vomma(k,x,tau,sigma,r))
    except:
        return float('NaN')

def put_iv(v, k, x, tau, r=0.0525):
    if v <= 0:
        return float('NaN')
    vcall = v + x - exp(-r*tau)*k
    return call_iv(vcall,k,x,tau,r)
