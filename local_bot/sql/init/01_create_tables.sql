-- Create database
CREATE DATABASE IF NOT EXISTS options;

-- Create greeks table
CREATE TABLE IF NOT EXISTS options.greeks (
    timestamp Int64,
    option_type String,
    expiration String,
    strike Int64,
    underlying String,
    data_type String,
    price Double,
    underlying_price Double,
    iv Double,
    mikhail_iv Double,
    delta Double,
    gamma Double,
    vega Double,
    theta Double
) ENGINE = MergeTree()
ORDER BY timestamp;
