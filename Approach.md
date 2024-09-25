# Approach to Validus Macro Strategy Case Study

## Data Filtering

The first step was to filter the provided data to identify the specific options being traded, based on the **out-of-the-money (OTM) percentage**. Since the problem assumes that the **forward price** of the underlying index equals the **spot price** today, and all the tasks involve **call options**, the strike price of the desired option is calculated as:

$
\text{Strike Price} = \text{Spot Price} \times (1 + \text{OTM percentage})
$

The specific option is then selected as the one with the closest strike price to this value.

---

## Mark-to-Market (MTM) Calculation

The **daily MTM value** for each option is calculated as the difference between the current market price (mid-point of bid and ask prices) and the traded price (the price at which the option was bought or sold). 
  - If we have **sold** the option, an **increasing** option price **decreases** our MTM value, while a **decreasing** option price **increases** our MTM value.
  - If we have **bought** the option, it is the reverse.

To find the **MTM of the entire portfolio**, we sum the MTM values of all the individual options in the portfolio for the corresponding day.

---

## Delta Calculation

Delta measures the **sensitivity of the option price** to movements in the underlying index price and is essentially the **first derivative** of the option price with respect to the index price.

### Finite Difference Method

The **finite difference method** is a numerical approximation of the derivative:

$
\Delta = \frac{\text{Option Price Change}}{\text{Underlying Price Change}}
$

In this method, the **first derivative** (delta) is approximated by dividing the change in the option price by the change in the index price. This method is implemented in the code due to its empirical nature.

However, one could use the **Black-Scholes model** to compute delta theoretically. For this, we first compute the **implied volatility** (\(\sigma\)), which is again calculated using the **Black-Scholes model**.

---

## Implied Volatility Calculation

To calculate the **implied volatility** ($\sigma$), the **Black-Scholes model** is used with the **Newton-Raphson method** for root-finding.

### Black-Scholes Option Pricing

The Black-Scholes model provides a formula for the price of a European call option based on inputs like the spot price of the underlying index, strike price, risk-free interest rate, implied volatility, and time to expiry. However, since we are given the market price of the option, we need to **invert** the Black-Scholes formula to solve for ($\sigma$).

Since the inverse of the Black-Scholes formula is not analytically tractable, I used the **Newton-Raphson method**, which iteratively updates the volatility estimate based on the gradients (vega) and the difference between the observed market price and the calculated price.

---

## Greek Comparison: Strategy I vs. Strategy II

### Task 6

Both Strategy I and Strategy II involve selling **1% OTM call options**, but they differ in their **time to expiration**:
- **Strategy I** involves a **one-month expiry** option.
- **Strategy II** involves a **two-month expiry** option.

So the key difference between the two strategies is how the Greeks behave due to the different expiry times.

### Delta

Delta measures the **sensitivity of the option price** to changes in the underlying asset's price.

**Strategy I (One-Month Option)** has less time to expiration, and the option's price is more sensitive to changes in the underlying price. Therefore, the delta is **higher** in magnitude because a small change in the underlying price can have a large impact on the option's value. While in **Strategy II (Two-Month Option)**, with more time until expiration, the longer time frame allows for more price movement without significantly affecting the option value.

### Gamma

Gamma measures the **rate of change of delta** with respect to the underlying price.

**Strategy I** has a **higher** Gamma. Because delta changes more rapidly as the option approaches expiration. The price of a shorter-term option is more sensitive to movements in the underlying asset. However, in **Strategy II**, the delta changes more slowly due to the additional time to expiration and Gamma is **lower**

### Theta

Theta measures the **sensitivity of the option price to the passage of time** (i.e., time decay).

**Strategy I** has a **higher** Theta, because the option is closer to expiration, and time decay accelerates as expiration approaches. The option loses value faster as time passes. But, in **Strategy II**, because there is more time until expiration, the time decay is slower due to more optionality value and Theta is **lower** 

### Vega

Vega measures the **sensitivity of the option price to changes in implied volatility**.

In **Strategy I**, Vega is **lower** because as an option approaches expiration, its price becomes less sensitive to changes in implied volatility. The option’s value is primarily driven by price movements and time decay. On teh other hand, in **Strategy II**, Vega is **higher** because the option has more time to expiration, so changes in implied volatility have a greater effect on its price.

---
