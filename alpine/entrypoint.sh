#!/bin/sh

su postgres -c 'pg_ctl start -D /var/lib/postgresql/data'
psql -U postgres --command="ALTER USER postgres PASSWORD 'postgres'"
# todo make it runtime
# currently pas a "top" helper
top