import os
import clickhouse_driver
from contextlib import contextmanager
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@contextmanager
def get_clickhouse_connection():
    host = os.getenv('CLICKHOUSE_HOST', 'localhost')
    port = int(os.getenv('CLICKHOUSE_PORT', 9000))
    user = os.getenv('CLICKHOUSE_USER', 'default')
    password = os.getenv('CLICKHOUSE_PASSWORD', '')
    client = clickhouse_driver.Client(host=host, port=port, user=user, password=password)
    # settings=get_clickhouse_settings())
    try:
        yield client
    finally:
        client.disconnect()


def get_clickhouse_settings():
    settings = {'max_query_size': 10000000}
    return settings


def get_sql(name, **kwargs):
    #print(**kwargs)
    with open(os.path.join(get_sql_path(), name)) as file:
        query = file.read()
        query_sql = query.format(**kwargs)

    with get_clickhouse_connection() as connection:
        res = connection.execute(query_sql, with_column_types=True)
        data, columns = res
        column_names = [column[0] for column in columns]
        df = pd.DataFrame(data, columns=column_names)

    return df



def get_sql_path():
    return os.path.join(get_path(), 'sql')


def get_path():
    return os.path.join('')

# def get_clickhouse_connection():
#     # https://confluence.exness.io/pages/viewpage.action?pageId=101752946
#     host = os.environ['CLICKHOUSE_HOST']
#     user = os.environ['CLICKHOUSE_USER']
#     password = os.environ['CLICKHOUSE_PASSWORD']
#     return clickhouse_driver.Client(host=host, port=29000, user=user, password=password,
#                                     settings=get_clickhouse_settings())

# %%