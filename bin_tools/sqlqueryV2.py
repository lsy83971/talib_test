import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import psycopg2 as pg
from pyhive import hive
from datetime import datetime,timedelta
import math
import sys
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import signal
import time
from io import StringIO

#import pymysql
    
def set_timeout(num):
    def wrap(func):
        def handle(signum, frame): # 收到信号 SIGALRM 后的回调函数，第一个参数是信号的数字，第二个参数是the interrupted stack frame.
          raise RuntimeError
        def to_do(*args, **kwargs):
          try:
              signal.signal(signal.SIGALRM, handle) # 设置信号和回调函数
              signal.alarm(num) # 设置 num 秒的闹钟
              logger.info('start alarm signal.')
              r = func(*args, **kwargs)
              logger.info('close alarm signal.')
              signal.alarm(0) # 关闭闹钟
              return r
          except BaseException as e:
              signal.alarm(0) # 关闭闹钟
              logger.info("err close alarm signal.")
              raise e
        return to_do
    return wrap

@set_timeout(3600) # 限时 3600 秒
def pgsql(sql):
    try:
        conn1 = pg.connect(dbname = "zbedw", user = "fwadmin", password = "ZBfwinst", 
                       port = "5432",host = "10.2.64.16", client_encoding = "UTF-8")
        logger.info(sql)
        df = pd.read_sql(sql,conn1)
        conn1.close()
        return df
    except BaseException as e:
        logger.info("sql raise error")
        try:
            conn1.close()
        except BaseException as e1:
            pass
        raise e
        
@set_timeout(3600) # 限时 3600 秒
def pgsql1(sql):
    try:
        conn1 = pg.connect(dbname = "zbedw", user = "dwadmin", password = "ZBdwinst", 
                       port = "5432",host = "10.2.64.16", client_encoding = "UTF-8")
        logger.info(sql)
        df = pd.read_sql(sql,conn1)
        conn1.close()
        return df
    except BaseException as e:
        logger.info("sql raise error")
        try:
            conn1.close()
        except BaseException as e1:
            pass
        raise e

### 测试用    
@set_timeout(3600) # 限时 3600 秒
def pgsql2(sql):
    try:
        conn1 = pg.connect(dbname = "zbedw", user = "gwadmin", password = "Zbgwinst", 
                           port = "5432",host = "10.2.64.16", client_encoding = "UTF-8")
        logger.info(sql)
        df = pd.read_sql(sql,conn1)
        conn1.close()
        return df
    except BaseException as e:
        logger.info("sql raise error")
        try:
            conn1.close()
        except BaseException as e1:
            pass
        raise e
    

### 测试用    
@set_timeout(3600) # 限时 3600 秒
def pgsql3(sql):
    try:
        conn1 = pg.connect(dbname = "zbedw", user = "cwadmin", password = "ZBBcwinst", 
                           port = "5432",host = "10.2.64.16", client_encoding = "UTF-8")
        logger.info(sql)
        df = pd.read_sql(sql,conn1)
        conn1.close()
        return df
    except BaseException as e:
        logger.info("sql raise error")
        try:
            conn1.close()
        except BaseException as e1:
            pass
        raise e
    

@set_timeout(3600) # 限时 3600 秒    
def hivesql(sql):
    try:
        conn = hive.Connection(host='10.2.64.1', port=10000, username='hive', password='hive', auth="CUSTOM",
                       database='default')
        df=pd.read_sql(sql,conn)
        conn.close()
        return df
    except BaseException as e:
        logger.info("sql raise error")
        try:
            conn1.close()
        except BaseException as e1:
            pass
        raise e
        
def creditApply(cols=["a.pid",
                      "a.apply_id", 
                      "a.channel_name",
                      "a.channel_id",
                      "a.channel_sec_id",                       
                      "a.id_card",
                      "a.apply_amt",
                      "a.apply_rate",
                      "a.apply_time",
                      "a.final_result",
                      "a.approve_state", 
                      "a.approve_amt", 
                      "a.approve_rate", 
],
                cond="",
                #apply_tab = "f_loan_contract_apply_info",
                apply_tab = "bi_base_apply_loan",
):
    """选取的指标可以包含"a.pid","a.apply_id","a.user_id","a.channel_name",
    "a.channel_id","a.phone","a.id_card","a.apply_amt","a.apply_rate","a.apply_time","a.final_result"...
    a.apply_amt可能是携程方提供的额度 一定大于0
    a.apply_rate是携程方提供的利率 在通过的时候为实际数 否则会等于0
    """
    sql="""select  {0}       
        from rskdm.{2} as a where 
    -- a.product_name='个人综合授信额度' and
    -- a.apply_type='授信')
    -- or (a.data_type='新信贷' and a.product_name='')
    ((a.data_type='新信贷' and a.channel_id in ('TC','10030ZYDLHB')) or 
    ((a.data_type in ('新信贷','信贷')) and a.channel_id not in ('TC','10030ZYDLHB'))
    ) and a.apply_type='授信'
    {1}""".format(",".join(cols),cond, apply_tab)
    
    if apply_tab in ["bi_base_apply_loan", "bi_base_apply_loan_zydapp"]:
        sql = sql.replace("channel_sec_id", "sub_cha_id")
        
    return hivesql(sql)

def loanApply(cols=["b.pid",   
                    "b.apply_id",
                    "b.id_card",
                    "b.channel_id",
                    "b.channel_sec_id",         
                    "b.apply_time",
                    "b.apply_amt",
                    "b.apply_rate",
                    "b.approve_rate",
                    "b.final_result", 
                    "b.approve_state", 
                    "b.approve_amt", 
                    "b.approve_rate", 
],
              cond="",
              #apply_tab = "f_loan_contract_apply_info"
              apply_tab = "bi_base_apply_loan",
):
    """
    cond例子 "where b.channel_id='ZBCASH'"
    选取的指标可以包含"b.pid","b.apply_id","b.id_card","b.channel_id","b.apply_time",
    "b.apply_amt","b.apply_rate","b.approve_rate","b.final_result"...
    """
    sql="""
    select         
    {0}
    from rskdm.{2} as b
    where         
    b.apply_type='用信' and
    ((b.data_type='新信贷' and b.channel_id in ('TC','10030ZYDLHB')) or 
    ((b.data_type in ('新信贷','信贷')) and b.channel_id not in ('TC','10030ZYDLHB')))
    {1}
    """.format(",".join(cols),cond, apply_tab)
    
    if apply_tab in ["bi_base_apply_loan", "bi_base_apply_loan_zydapp"]:
        sql = sql.replace("channel_sec_id", "sub_cha_id")
    return hivesql(sql)

def loanInfo(cols=["b.pid",
                   "b.channel_id",
                   "b.channel_sec_id",
                   "b.apply_id", 
                   "b.id_card",
                   "c.loan_no",
                   "c.grant_date",
                   "c.init_principal",
                   "c.init_term",
                   "c.paid_out_date",
                   "c.left_principal",
                   "c.real_overdue_days",
                   "c.overdue_days",
                   "c.overdue_principal",
                   "c.loan_status",
                   "c.float_rate",
                   "c.interest_rate",
                   "c.creditloanno", 
                   "c.due_type"],cond1=None,cond2="",
             #apply_tab = "f_loan_contract_apply_info"
             apply_tab = "bi_base_apply_loan"
):
    """
    cond1包含etl_dt信息 可以"='2019-08-08'" 或者 "in ('2019-08-08','2019-08-08')"
    cond2可以包含任意信息 例如"and b.channel_id='ZBCASH'"
    cols可选"b.pid", "b.channel_id","c.loan_no","c.grant_date","c.init_principal",
    "b.id_card","b.apply_time","b.apply_amt","b.apply_rate","b.approve_rate",
    "c.init_term","c.paid_out_date","c.left_principal","c.real_overdue_days","c.overdue_days",
    "c.overdue_principal","c.loan_status","c.due_type"...
    """
    if not isinstance(cond1,str):
        DT=(datetime.now()-timedelta(2)).strftime("%Y-%m-%d")        
        cond1="='"+DT+"'"
    sql_loan = """select 
    {0}
    from rskdm.{3} b
    inner join rskdm.bi_base_loan_info as c
    on  c.contract_no=b.serialno and
        b.serialno <> '' and
        ((b.data_type='新信贷' and b.channel_id in ('TC','10030ZYDLHB')) or 
        ((b.data_type in ('新信贷','信贷')) and b.channel_id not in ('TC','10030ZYDLHB'))) and
        b.product_name='众易贷' and
        b.apply_type='用信'
    where c.data_type='信贷' and 
          c.etl_dt {1} 
    {2}
            """.format(",".join(cols),cond1,cond2, apply_tab)
    if apply_tab == "bi_base_apply_loan":
        sql_loan = sql_loan.replace("channel_sec_id", "sub_cha_id")
    return hivesql(sql_loan)

def repayInfo(cols=[
        "b.apply_id", 
        "d.loan_no",
        "d.repay_plan_status",
        "d.repay_type",
        "d.last_repay_time",
        "d.schedule_term",
        "d.init_repay_date"
],
              cond1="",
              DT="",
              #apply_tab = "f_loan_contract_apply_info",
              apply_tab = "bi_base_apply_loan",
              repay_tab = "bi_repay_plan"):
    """
    cond1包含etl_dt信息 可以"='2019-08-08'" 或者 "in ('2019-08-08','2019-08-08')"
    cond2可以包含任意信息 例如"and b.channel_id='ZBCASH'"
    cols可选 b. c. d. 之中的一切...
    """

    if DT=="":
        DT=(datetime.now()-timedelta(2)).strftime("%Y-%m-%d")        
    sql_repay = """select 
    {0}
    from 
    rskdm.{3} b
    inner join 
    rskdm.bi_base_loan_info c
    on  c.contract_no=b.serialno and
        b.serialno <> '' and
        b.product_name='众易贷' and
        b.apply_type='用信' and
        ((b.data_type='新信贷' and b.channel_id in ('TC','10030ZYDLHB')) or 
        ((b.data_type in ('新信贷','信贷')) and b.channel_id not in ('TC','10030ZYDLHB')))
    inner join 
    rskdm.{4} d        
    on
    d.loan_no=c.loan_no
    and c.etl_dt='{1}' 
    where 1=1 {2}
    """.format(",".join(cols), DT, cond1, apply_tab, repay_tab)
    
    if apply_tab == "bi_base_apply_loan":
        sql_repay = sql_repay.replace("channel_sec_id", "sub_cha_id")
        
    return hivesql(sql_repay)

## 可能连续几次的提前还款
## repaytype为空 等价于 未还款
## apply_loan有重复

def tdData(date_begin,date_end,table_name="es_third_service_data",channel="all",link="1"):
    """
    table_name三种选择
    es_req_params
    es_resp_params
    es_third_service_data
    """
    # es_req_params
    date_begin1=date_begin
    date_end1=date_end
    if len(date_begin)==8:
        date_begin="-".join([date_begin[:4],date_begin[4:6],date_begin[6:8]])
    if len(date_end)==8:
        date_end="-".join([date_end[:4],date_end[4:6],date_end[6:8]])  
    date_end=(datetime.strptime(date_end,"%Y-%m-%d")+timedelta(1)).strftime("%Y-%m-%d")
    if isinstance(channel,str):
        if channel=="all":
            channel_info=""
        else:
            channel_info=" and b.product_code='{0}' ".format(channel)
    else:
        if isinstance(channel,list):
            channel_info=" and b.product_code in ('{0}') ".format("','".join(channel))
            
    if link=="all":
        link_info=""
    else:
        link_info=" and b.link='{0}'".format(link)
    sql="""select a.field_name,a.field_value,b.entry_id from 
    risktd.{0} as a 
    inner join risktd.es_base_info as b 
    on a.token=b.token 
    where b.record_time>='{1}' and b.record_time<='{2}'
    {3} {4}""".format(table_name,date_begin,date_end,channel_info,link_info)
    logger.info("开始读取...")
    _df=pgsql1(sql)
    logger.info("数据读取成功")
    _df.drop_duplicates(["entry_id","field_name"],inplace=True)
    return _df.pivot(index="entry_id",columns="field_name",values="field_value")

def param_channel(date_begin = None,date_end = None,table_name="es_third_service_data",channel="ZYDLHPLOAN",link="2", ret_sql = False, fields = None, add = ""):
    """
    table_name三种选择
    es_req_params
    es_resp_params
    es_third_service_data
    """
    # es_req_params
    if date_begin is not None:
        date_begin = pd.to_datetime(date_begin).strftime("%Y-%m-%d")
    if date_end is not None:
        date_end = pd.to_datetime(date_end).strftime("%Y-%m-%d")

    if date_begin is None:
        date_begin_word = ""
    else:
        date_begin_word = " and b.record_time>='{0}' ". format(date_begin)

    if date_end is None:
        date_end_word = ""
    else:
        date_end_word = " and b.record_time<='{0}' ". format(date_end)
            
    if isinstance(channel,str):
        if channel=="all":
            channel_info=""
        else:
            channel_info=" and b.product_code='{0}' ".format(channel)
    else:
        if isinstance(channel,list):
            channel_info=" and b.product_code in ('{0}') ".format("','".join(channel))
            
    if link=="all":
        link_info=""
    else:
        link_info=" and b.link='{0}'".format(link)

    if fields is None:
        fields_info = ""
    else:
        if isinstance(fields, str):
            fields_info = " and a.field_name='{0}'". format(fields)
        elif isinstance(fields, list):
            fields_info = " and a.field_name in ('{0}')". format("','". join(fields))
        else:
            raise("GG")

    sql="""select a.field_name,a.field_value,b.entry_id from 
    risktd.{0} as a 
    inner join risktd.es_base_info as b 
    on a.token=b.token 
    where 1=1 {1} {2}
    {3} {4} {5} {6}""".format(table_name,date_begin_word,date_end_word,channel_info,link_info, fields_info, add)
    if ret_sql == True:
        return sql
    logger.info("开始读取...")
    _df=pgsql1(sql)
    logger.info("数据读取成功")
    _df.drop_duplicates(["entry_id","field_name"],inplace=True)
    return _df.pivot(index="entry_id",columns="field_name",values="field_value")

def param_ctrip():
    zydyq_scai_ride_req=pgsql("select * from ods.zydyq_scai_ride_req").set_index("seq_no")
    zydyq_scai_zbank_rule_req=pgsql("select * from ods.zydyq_scai_zbank_rule_req").set_index("seq_no")
    zydyq_scai_ride_resp=pgsql("select * from ods.zydyq_scai_ride_resp").set_index("seq_no")
    zydyq_scai_zbank_rule_resp=pgsql("select * from ods.zydyq_scai_zbank_rule_resp").set_index("seq_no")
    
    zydyq_scai_ride_req.columns="ride_req."+pd.Series(zydyq_scai_ride_req.columns).apply(lambda x:x.split(".")[-1])
    zydyq_scai_zbank_rule_req.columns="rule_req."+pd.Series(zydyq_scai_zbank_rule_req.columns).apply(lambda x:x.split(".")[-1])
    zydyq_scai_ride_resp.columns="ride_resp."+pd.Series(zydyq_scai_ride_resp.columns).apply(lambda x:x.split(".")[-1])
    zydyq_scai_zbank_rule_resp.columns="rule_resp."+pd.Series(zydyq_scai_zbank_rule_resp.columns).apply(lambda x:x.split(".")[ - 1])
    
    _df=zydyq_scai_zbank_rule_req.merge(zydyq_scai_ride_req,left_index=True,right_index=True)
    _df=_df.merge(zydyq_scai_ride_resp,left_index=True,right_index=True)
    _df=_df.merge(zydyq_scai_zbank_rule_resp,left_index=True,right_index=True)

    return _df

def rod(channel, dt):
    dt = str(pd.to_datetime(dt).date())
    sql = """
    select a.*,b.apply_id from
    (select
     duebill_num,
     period_seq_num,
     paid_dt,
     payable_dt,
     repay_dt,
     row_number()
     over
     (
      partition by
      duebill_num, period_seq_num
      order by
      paid_dt desc,paid_princp desc,update_dt desc
     ) rn
     from
      (
       select
       paid_princp,
       update_dt,
       duebill_num,
       period_seq_num,
       case
       when paid_dt is NULL then ''
       when paid_dt = "null" then ''
       when paid_dt = "NULL" then ''
       else paid_dt
       end as paid_dt,
       payable_dt,
       repay_dt
       from
       rskdm.cha_roverdue
      ) c
    ) a
    inner join rskdm.bi_base_loan_info as b
    on a.duebill_num = b.loan_no
    where
    b.etl_dt='{0}' and
    b.channel_id='{1}' and
    a.rn=1
    """. format(dt, channel)
    y_channel = hivesql(sql)
    colr = \
    {
        "a.paid_dt": "d.last_repay_time",
        "a.payable_dt": "d.init_repay_date",
        "a.period_seq_num": "d.schedule_term", 
        "a.duebill_num": "d.loan_no"
    }
    y_channel.rename(columns = colr, inplace = True)
    return y_channel

def pgCol(name):
    sql="""
    Select a.attnum,(select description from pg_catalog.pg_description where objoid=a.attrelid and objsubid=a.attnum) as descript ,a.attname,pg_catalog.format_type(a.atttypid,a.atttypmod) as data_type,(select relname from pg_class where oid=a.attrelid ) as table_name from pg_catalog.pg_attribute a where 1=1 and a.attrelid in (select oid from pg_class where relname like '%{0}%' ) and a.attnum>0 and not a.attisdropped order by a.attnum;
    """.format(name)
    return pgsql(sql)




def df_to_mpp(df, mpp_tbname, mpp_columns, mpp_user='gwadmin'):
    target_df=df[mpp_columns]
    if mpp_user=='gwadmin':
        conn = pg.connect(host='10.2.64.16', user='gwadmin', password='Zbgwinst',
                          dbname='zbedw', port='5432', client_encoding='utf-8')
    elif mpp_user=='cwadmin':
        conn = pg.connect(host='10.2.64.16', user='cwadmin', password='ZBBcwinst',
                          dbname='zbedw', port='5432', client_encoding='utf-8')
    else:
        conn = pg.connect(host='10.2.64.16', user='fwadmin', password='ZBfwinst',
                          dbname='zbedw', port='5432', client_encoding='utf-8')
        
    cur = conn.cursor()
    try:
        cur.copy_from(StringIO(target_df.to_csv(sep='\t', index=False, header=False)),
                      mpp_tbname, null='', columns=mpp_columns)
        conn.commit()
        cur.close()
    except Exception as e:
        print(e)
    except BaseException as be:
        print(be)
    finally:
        conn.close()

def to_mpp(self, tbl_name):
    df = self
    mpp_tbname = "tmp." + tbl_name
    mpp_columns = df.columns.tolist()
    df_to_mpp(df, mpp_tbname, mpp_columns)
    
pd.DataFrame.to_mpp = to_mpp
    
    





