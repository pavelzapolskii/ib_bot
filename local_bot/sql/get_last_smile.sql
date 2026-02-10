WITH (
    SELECT max(timestamp) - {hours}*60*60 FROM options.greeks 
   where option_type='{option_type}' 
   AND underlying='{underlying}' 
   AND expiration='{expiration}'
) AS prev_timestamp 
select strike, data_type, iv, mikhail_iv, prev_timestamp, timestamp
    from options.greeks
where option_type='{option_type}' 
    AND underlying='{underlying}' 
    AND expiration='{expiration}'
    AND timestamp >= (toDateTime(prev_timestamp) - INTERVAL 50 SECOND)
    AND timestamp <= (toDateTime(prev_timestamp))
