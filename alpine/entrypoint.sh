#!/bin/sh
su postgres -c 'pg_ctl start -D /var/lib/postgresql/data'
# todo make it runtime
# currently pas a "top" helper
top