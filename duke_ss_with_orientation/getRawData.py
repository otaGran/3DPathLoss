import psycopg2
from decimal import Decimal
import pandas as pd


def get_data(postgreSQL_execute):
    connection = psycopg2.connect(database='postgres', user='postgres',
                                  password='59^N0PFxsddJ', host='35.170.61.170', port=55431)
    cursor = connection.cursor()
    cursor.execute(postgreSQL_execute)

    rows = cursor.fetchall()
    return rows, cursor
# {
# "db_host": "35.170.61.170",
# "db_port": 55431,
# "db_name": "postgres",
# "db_user": "postgres",
# "db_password": "59^N0PFxsddJ"
# }


def rows_to_dict(rowsss, cursor):
    columns = [desc[0] for desc in cursor.description]
    result = []
    for rrr in rowsss:
        result.append(dict(zip(columns, rrr)))
    return result


def combine_dicts_to_dict_of_lists(dict_list):
    result = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key not in result:
                result[key] = []
            result[key].append(value)
    for key in result.keys():
        if isinstance(result[key][0], Decimal):
            for iii, item in enumerate(result[key]):
                result[key][iii] = float(item)
    return result


def return_pd(postgreSQL_execute="SELECT AVG(latitude) as latitude, AVG(longitude) as longitude, AVG(rssi) as rssi, "
                                 "AVG(azimuth) as azimuth FROM \"SigCapDetails\" WHERE \"mPci\"=20 group by \"mTimeStamp\""):
    rows, cursor = get_data(postgreSQL_execute)
    dict_lst = combine_dicts_to_dict_of_lists(rows_to_dict(rowsss=rows, cursor=cursor))
    df_SigCapDetails = pd.DataFrame(dict_lst)
    return df_SigCapDetails
