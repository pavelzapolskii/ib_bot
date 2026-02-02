-- Get ATM (50 delta) strikes for each symbol/expiration
-- Uses the most recent data and finds strike closest to 0.50 delta for calls
SELECT
    underlying,
    expiration,
    strike,
    delta,
    iv,
    underlying_price,
    timestamp
FROM (
    SELECT
        underlying,
        expiration,
        strike,
        delta,
        iv,
        underlying_price,
        timestamp,
        ROW_NUMBER() OVER (PARTITION BY underlying, expiration ORDER BY abs(delta - 0.5) ASC, timestamp DESC) as rn
    FROM options.greeks
    WHERE option_type = 'C'
      AND delta > 0
)
WHERE rn = 1
ORDER BY underlying, expiration
