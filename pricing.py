import numpy as np
from scipy.stats import norm

def put_payoff(S, K):
    """Payoff of a put option at expiry: max(K - S, 0)."""
    return np.maximum(K - S, 0)

def call_payoff(S, K):
    """Payoff of a call option at expiry: max(S - K, 0)."""   
    return np.maximum(S - K, 0)

def one_step_call(S0, K, u, d, r, T):
    """Price a European call with a one-step binomial tree (up factor u, down factor d)."""
    S_u = S0 * u
    S_d = S0 * d
    f_u = call_payoff(S_u, K)
    f_d = call_payoff(S_d, K)
    p = ((np.exp(r * T)) - d) / (u - d)
    f = (np.exp(-r * T)) * (p * f_u + (1 - p) * f_d)
    return f 

def binomial_call(S0, K, r, sigma, T, n):
    """Price a European call with an n-step binomial tree (u = e^(sigma*sqrt(dt)), d = 1/u)."""
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    j = np.arange(n + 1)
    S_t = S0 * (u ** j) * (d ** (n - j))
    f = call_payoff(S_t, K)
    p = ((np.exp(r * dt)) - d) / (u - d)
    disc = np.exp(-r * dt)
    for _ in range(n):
        f = disc * (p * f[1:] + (1 - p) * f[:-1])
    return f[0]

def binomial_american_put(S0, K, r, sigma, T, n):
    """Price an American put with an n-step binomial tree, checking early exercise at every node."""
    dt = T / n  
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    j = np.arange(n + 1)
    S_t = S0 * (u ** j) * (d ** (n - j))
    f = put_payoff(S_t, K)
    p = ((np.exp(r * dt)) - d) / (u - d)
    disc = np.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        f = disc * (p * f[1:] + (1 - p) * f[:-1])
        j = np.arange(i + 1)
        S_i = S0 * (u ** j) * (d ** (i - j))
        f = np.maximum(f, put_payoff(S_i, K))
    return f[0]

def monte_carlo_simulation(S0, K, r, sigma, T, n, seed, option_type):
    """Price a European call or put by Monte Carlo with n paths; returns (price, standard error)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    S_t = S0 * np.exp((r - ((sigma ** 2) / 2)) * T + sigma * np.sqrt(T) * Z)
    disc = np.exp(-r * T)
    if option_type == "call":
        payoffs = call_payoff(S_t, K) * disc
    elif option_type == "put":
        payoffs = put_payoff(S_t, K) * disc
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    f = np.mean(payoffs)
    se = np.std(payoffs) / np.sqrt(n)
    return f, se

def black_scholes(S, K, T, r, sigma, option_type):
    """Price a European call or put with the Black-Scholes formula."""
    d1 = (np.log(S / K) + ((r + (sigma ** 2 / 2)) * T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call
    elif option_type == "put":
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return put
    else:
        raise ValueError("Option type must be 'call' or 'put'")

def delta(S, K, T, r, sigma, h):
    """Call delta by central differences: change in price per £1 change in S."""
    price_up = black_scholes(S + h, K, T, r, sigma, "call")
    price_down = black_scholes(S - h, K, T, r, sigma, "call")
    return (price_up - price_down) / (2 * h)

def delta_formula(S, K, T, r, sigma, option_type):
    """Delta from the Black-Scholes formula: N(d1) for a call, N(d1) - 1 for a put."""
    d1 = (np.log(S/K) + ((r + (sigma ** 2 / 2)) * T)) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return -norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either call or put")

def vega(S, K, T, r, sigma, h):
    """Call vega by central differences, per 1.00 change in sigma (divide by 100 for per 1%)."""
    price_up = black_scholes(S, K, T, r, sigma + h, "call")
    price_down = black_scholes(S, K, T, r, sigma - h, "call")
    return (price_up - price_down) / (2 * h)

def theta(S, K, T, r, sigma, h):
    """Call theta by central differences, in £ per year (negative = value lost as time passes)."""
    price_up = black_scholes(S, K, T + h, r, sigma, "call")
    price_down = black_scholes(S, K, T - h, r, sigma, "call")
    return -(price_up - price_down) / (2 * h)

def gamma(S, K, T, r, sigma, h):
    """Call gamma by central differences: change in delta per £1 change in S."""
    delta_up = delta(S + h, K, T, r, sigma, h)
    delta_down = delta(S - h, K, T, r, sigma, h)
    return (delta_up - delta_down) / (2 * h)

def implied_vol(market_price, S, K, T, r):
    """Implied volatility of a European call by bisection, searching sigma between 0.01 and 2."""
    low = 0.01
    high = 2
    while high - low > 0.000001:
        sigma = (high + low) / 2
        if black_scholes(S, K, T, r, sigma, "call") > market_price:
            high = sigma
        else:
            low = sigma
    return sigma  