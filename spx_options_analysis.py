import pandas as pd
import numpy as np
from scipy.stats import norm

# Strategy creation function for various buy/sell options
def create_strategy(spx_df, trade_date, options_actions):
    """
    Function to create a strategy and calculate MTM and delta daily values for the corresponding portfolio.

    Inputs:
    spx_df (DataFrame): The SPX dataset.
    trade_date (str): The trade start date.
    options_actions (list of dicts): A list of actions specifying the side (buy/sell), expiry length (in months),
                                    OTM percentage.

    Returns:
    total_mtm (DataFrame): Portfolio MTM values for the strategy.
    total_delta (DataFrame): Portfolio delta values for the strategy.
    options_traded (list): Specific options traded in the strategy.
    """

    options_traded= []
    total_delta = pd.DataFrame()
    total_mtm = pd.DataFrame()

    # Add new columns to spx_df to hold MTM and delta values
    spx_df['mtm'] = 0.0
    spx_df['delta'] = 0.0

    trade_date = pd.to_datetime(trade_date)

    # Process each action in the strategy
    for action in options_actions:
        otm_percentage = action['otm_percentage']
        expiry_length = action['expiry_length']
        action_side = action['side']

        # Calculate the OTM strike price
        spx_spot = spx_df[spx_df['date'] == trade_date]['adjusted close'].unique()[0]
        otm_strike = spx_spot * (1 + otm_percentage / 100)

        # Get expiry date for the given expiry length
        all_expiry_dates = spx_df[(spx_df['date'] == trade_date) & (spx_df['call/put'] == 'C')][
            'expiration'].sort_values().unique()
        if len(all_expiry_dates) < expiry_length:
            raise ValueError(f"Not enough expiry dates available for {expiry_length}-month expiry.")
        expiry_date = all_expiry_dates[expiry_length - 1]

        # Find the closest OTM call option for that expiry
        otm_call_option = spx_df[
            (spx_df['date'] == trade_date) &
            (spx_df['call/put'] == 'C') &
            (spx_df['expiration'] == expiry_date) &
            (spx_df['strike'] >= otm_strike)
        ].sort_values('strike').iloc[0]

        options_traded.append(otm_call_option)

        # Relevant rows for the option traded
        option_indices = spx_df['option symbol'] == otm_call_option['option symbol']

        # Calculate MTM for each day
        spx_df.loc[option_indices, 'mtm'] = spx_df.loc[option_indices].apply(
            lambda row: calculate_mtm(row, otm_call_option, action_side), axis=1)
        total_mtm = pd.concat([total_mtm, spx_df.loc[option_indices, ['date', 'mtm']]])

        # Calculate daily delta using the finite difference method
        spx_df.loc[option_indices, 'delta'] = spx_df.loc[option_indices, 'mid price'].diff() / spx_df.loc[
            option_indices, 'adjusted close'].diff()

        # Adjust delta for buy/sell actions
        spx_df.loc[option_indices, 'delta'] = spx_df.loc[option_indices, 'delta'].apply(
            lambda delta: delta if action_side == 'buy' else -delta)

        total_delta = pd.concat([total_delta, spx_df.loc[option_indices, ['date', 'delta']]])

    # Group by date and sum all mtms and deltas for each day
    total_mtm = total_mtm.groupby('date')['mtm'].sum().reset_index()
    total_delta = total_delta.groupby('date')['delta'].sum().reset_index()

    return total_mtm, total_delta, options_traded


# Function to calculate MTM for a Call option.
def calculate_mtm(curr_day, traded_option, action_side):
    """
    Calculate the MTM (Mark-to-Market) for a specific option on a given day.

    Inputs:
    curr_day (Series): Row data for the current trading day.
    traded_option (Series): Details of the traded option.
    action_side (str): 'buy' or 'sell' action.

    Returns:
    float: MTM value for the option on the current day.
    """
    trade_date = traded_option['date']
    expiry_date = traded_option['expiration']
    strike = traded_option['strike']
    trade_price = traded_option['bid'] if action_side == 'sell' else traded_option['ask']

    if curr_day['date'] <= trade_date or curr_day['date'] > expiry_date:
        return 0

    if curr_day['date'] < expiry_date:
        market_price = curr_day['mid price']
        return trade_price - market_price if action_side == 'sell' else market_price - trade_price

    option_value = max(0, curr_day['adjusted close'] - strike)
    return trade_price - option_value if action_side == 'sell' else option_value - trade_price


# Functions for implied volatility and Black-Scholes calculation

# Calculate d1 and d2 for Black-Scholes model
def calculate_d1_d2(S, K, T, sigma):
    """
    Inputs:
    S (float): Current SPX price.
    K (float): Strike price.
    T (float): Time to expiration in years.
    sigma (float): Implied Volatility.

    Returns:
    tuple: d1 and d2 values.
    """
    if T <= 0:
        return np.nan, np.nan
    sigma = max(sigma, 1e-10)  # Prevent division by zero in sigma
    d1 = (np.log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# Black-Scholes Call option price using d1 and d2
def black_scholes_call(S, K, T, sigma):
    if T <= 0:
        return max(S - K, 0)
    d1, d2 = calculate_d1_d2(S, K, T, sigma)
    return S * norm.cdf(d1) - K * norm.cdf(d2)


# Calculate vega (sensitivity to volatility)
def vega(S, K, T, sigma):
    if T <= 0:
        return 0.0  # No vega at expiration

    d1, _ = calculate_d1_d2(S, K, T, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)

# Newton-Raphson method to find implied volatility (root finding)
def implied_volatility_call_newton(market_price, S, K, T, tol=1e-6, max_iter=100):
    sigma = 0.5  # Initial guess for IV
    for i in range(max_iter):
        diff = black_scholes_call(S, K, T, sigma) - market_price
        if abs(diff) < tol:  # Check if within the tolerance level
            return sigma
        vega_value = vega(S, K, T, sigma)
        if vega_value == 0:  # Avoid division by zero
            return np.nan
        sigma -= diff / vega_value  # Update sigma using Newton-Raphson method
    return sigma  # the best estimate after max iterations


# Implied volatility calculation for specific traded option
def calculate_iv(option_data):
    """
    Inputs:
    option_data (Series): Info of the specific option traded

    Returns:
    DataFrame: A DataFrame with the date and implied volatility for the specific traded option.
    """
    option_df = spx_df[(spx_df['option symbol'] == option_data['option symbol'])].copy()
    option_df['IV'] = option_df.apply(lambda row: implied_volatility_call_newton(
        row['mid price'], row['adjusted close'], row['strike'],
        (row['expiration'] - row['date']).days / 365
    ), axis=1)
    return option_df[['date', 'IV']]

# Load the S&P500 dataset and apply the functions
spx_df = pd.read_csv('SPX_Monthly_Option_data_300121_300421.csv')
spx_df = spx_df.sort_values('date')
spx_df['date'] = pd.to_datetime(spx_df['date'], format='%m/%d/%Y')
spx_df['expiration'] = pd.to_datetime(spx_df['expiration'], format='%m/%d/%Y')
spx_df['mid price'] = (spx_df['bid'] + spx_df['ask']) / 2

# Task 1-Strategy I: Sell 1% OTM one-month call option on Feb 1st, 2021
options_actions_strategy_1 = [{'side': 'sell', 'expiry_length': 1, 'otm_percentage': 1}]
mtm_strategy_1, _, option_strategy_1 = create_strategy(spx_df, '2021-02-01', options_actions_strategy_1)
mtm_strategy_1['option symbol'] = option_strategy_1[0]['option symbol']
mtm_strategy_1.to_csv('strategy1_mtm.csv', index=False)

# Task 2-Strategy II: Sell 1% OTM two-month call option on Feb 1st, 2021
options_actions_strategy_2 = [{'side': 'sell', 'expiry_length': 2, 'otm_percentage': 1}]
mtm_strategy_2, _, option_strategy_2= create_strategy(spx_df, '2021-02-01', options_actions_strategy_2)
mtm_strategy_2['option symbol'] = option_strategy_2[0]['option symbol']
mtm_strategy_2.to_csv('strategy2_mtm.csv', index=False)

# Task 3-Strategy III: Sell 1% OTM two-month call option and buy 2% OTM two-month call option on Feb 1st, 2021
options_actions_strategy_3 = [{'side': 'sell', 'expiry_length': 2, 'otm_percentage': 1},
                                {'side': 'buy', 'expiry_length': 2, 'otm_percentage': 2}]
mtm_strategy_3, delta_strategy_3, option_strategy_3 = create_strategy(spx_df, '2021-02-01', options_actions_strategy_3)
mtm_strategy_3['option symbol'] = option_strategy_3[0]['option symbol'] + "+" + option_strategy_3[1]['option symbol']
mtm_strategy_3.to_csv('strategy3_mtm.csv', index=False)

# Task 4: Delta for strategy III
delta_strategy_3['option symbol'] = option_strategy_3[0]['option symbol'] + "+" + option_strategy_3[1]['option symbol']
delta_strategy_3.to_csv('strategy3_delta.csv', index=False)

# Task 5: Calculate implied volatility for Strategy II
iv_strategy2 = calculate_iv(option_strategy_2[0])
iv_strategy2['option symbol'] = option_strategy_2[0]['option symbol']
iv_strategy2.to_csv('strategy2_ivolatility.csv', index=False)