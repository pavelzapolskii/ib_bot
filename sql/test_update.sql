insert into options.greeks 
    (timestamp, option_type, expiration, strike, underlying, data_type, price, underlying_price, iv, mikhail_iv, delta, gamma, vega, theta)
    values 
    ({timestamp}, '{option_type}', '{expiration}', {strike}, '{underlying}', '{data_type}', {price}, {underlying_price}, {iv}, {mikhail_iv}, {delta}, {gamma}, {vega}, {theta}
);