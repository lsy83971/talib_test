#!/bin/bash
rm -f /home/lishiyu/talib_test/data_process/sql_func/custom_function.xml
rm -f /home/lishiyu/talib_test/data_process/sql_func/lsy_exch_detail.py
cp /etc/clickhouse-server/custom_function.xml /home/lishiyu/talib_test/data_process/sql_func/
cp /var/lib/clickhouse/user_scripts/lsy_exch_detail.py /home/lishiyu/talib_test/data_process/sql_func/
git add *
git commit -m 'add'
git push'
