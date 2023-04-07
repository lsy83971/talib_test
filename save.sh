#!/bin/bash
rm -f /home/lishiyu/talib_test/data_process/sql_func/custom_function.xml
cp -f /etc/clickhouse-server/custom_function.xml /home/lishiyu/talib_test/data_process/sql_func/
git add *
git commit -m 'add'
git push'
