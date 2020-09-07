#  Created at 2019/4/4                     
#  Aothor: Ethan Lee                       
#  E-mail: ethan2lee.coder@gmail.com        
import pandas as pd
from sqlalchemy import create_engine
dirfolder = './data/'


def var_trans1(df,column_name, rate=0.05):
    """
    將類別種類數量大的類別型屬性合併與轉換
    輸入：
    var_series: pandas.Series: 要轉換的類別屬性。
    rete: 每個屬性的最小佔比。

    輸出：
    pandas.Series: 轉換後的類別屬性
    """

    var_series=df[column_name]
    total_counts = int(len(var_series) * (1 + rate / 2))  # var_series 不能包含NA, rate/2 為
    value_counts = var_series.value_counts(ascending=True).reset_index()
    value_counts.columns = [column_name, 'count']
    value_counts['cumsum'] = value_counts['count'].cumsum()
    value_counts['group'] = value_counts['cumsum'].apply(lambda item: \
                                                             (item / total_counts) // rate)
    total_groups = value_counts['group'].max() + 1
    value_counts['group'] = total_groups - value_counts['group']
    value_counts['group'] = value_counts['group'].round().astype(int).astype(str)
    dict=value_counts[[column_name, 'group']].to_dict()
    tmpdf = pd.DataFrame(var_series)
    tmpdf.columns = [column_name]
    tmpdf = tmpdf.merge(value_counts[[column_name, 'group']], how='inner', on=[column_name])
    df=tmpdf['group'].copy()
    return df,dict


def var_trans(df,column_name):
    """將類別種類數量大的類別型屬性合併與轉換"""
    var_series = df[column_name].copy()
    value_counts = var_series.value_counts()
    if len(value_counts) > 50:
        return var_trans1(df,column_name, rate=0.05)
    if len(value_counts) > 10:
        return var_trans1(df,column_name, rate=0.1)
    return var_series

def category_reduction(df,column_name):
    dict = var_trans(df, column_name)[1]
    df[column_name]=var_trans(df,column_name)[0]
    return dict


def flight_type_transfrom(AODB):
    #飛機類型特性表
    c_aircraft = pd.read_excel(dirfolder+'C_aircraft_v2.4_20190228.xlsx')
    columns = ['Manufactuer', 'Model', 'maxPassengers', 'Speed(mph)', 'WTC', \
            'Service_Ceiling(ft)', 'Range(NMi)', 'OEW(lbs)', 'MTOW(lbs)', 'Wing_Span(ft)', \
            'Wing_Area(ft2)', 'Length(ft)', 'Height(ft)']
    c_aircraft = c_aircraft[columns]
    c_aircraft[['Manufactuer', 'Model']] = c_aircraft[['Manufactuer', 'Model']].astype(str)

    m_aircraft = pd.read_excel(dirfolder+'M_aircraft_v2_20190305.xlsx')
    columns = ['AC_AIRCRAFT', 'AT_AIRCRAFT_TYPE', 'AT_SUBTYPE', 'Manufactuer', 'Model']
    m_aircraft = m_aircraft[columns]
    m_aircraft = m_aircraft.astype(str)

    AODB[['AC_AIRCRAFT', 'AT_AIRCRAFT_TYPE', 'AT_SUBTYPE']] = AODB[['AC_AIRCRAFT', \
        'AT_AIRCRAFT_TYPE', 'AT_SUBTYPE']].astype(str)
    AODB = AODB.merge(m_aircraft, how='left', on=['AC_AIRCRAFT', 'AT_AIRCRAFT_TYPE', 'AT_SUBTYPE'])
    AODB = AODB.merge(c_aircraft, how='left', on=['Manufactuer', 'Model'])

    AODB = AODB.drop(['AC_AIRCRAFT', 'AT_AIRCRAFT_TYPE', 'AT_SUBTYPE', 'Manufactuer', \
                      'Model'], axis=1) # shape:(329320, 36)
    del c_aircraft, m_aircraft, columns
    return AODB


def  server_transform(AODB):
    #服務類型特性表
    c_service = pd.read_excel(dirfolder+'C_service_v1_20190312.xlsx')
    columns = ['ST_SERVICE_TYPE', 'ST_Scheduled', 'ST_Passenger', 'ST_Cargo', 'ST_Mail', \
               'ST_Charter']
    c_service = c_service[columns]

    AODB = AODB.merge(c_service, how='left', on=['ST_SERVICE_TYPE'])

    AODB = AODB.drop(['ST_SERVICE_TYPE'], axis=1) # shape:(329320, 44)
    del c_service, columns
    return AODB

def test():
    pass


