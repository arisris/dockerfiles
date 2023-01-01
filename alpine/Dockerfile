FROM alpine:3.17.0

LABEL maintainer="Aris Riswanto <sksnet.id@gmail.com>"

RUN apk update && mkdir /web && \
  apk add micro && \
  apk add php81 php81-pdo php81-pdo_pgsql php81-pdo_mysql php81-pdo_sqlite php81-bcmath php81-ctype php81-fileinfo php81-mbstring php81-bz2 php81-posix php81-dom php81-ftp php81-session php81-tokenizer php81-xml php81-curl php81-brotli && \
  apk add postgresql15 && \
  mkdir /run/postgresql && \
  chown postgres:postgres /run/postgresql/

USER postgres

WORKDIR /var/lib/postgresql

RUN mkdir /var/lib/postgresql/data && \
  chmod 0700 /var/lib/postgresql/data && \
  initdb -D /var/lib/postgresql/data && \
  echo "host all all 0.0.0.0/0 md5" >> /var/lib/postgresql/data/pg_hba.conf && \
  echo "listen_addresses='*'" >> /var/lib/postgresql/data/postgresql.conf

USER root

WORKDIR /web

COPY ./entrypoint.sh /web/entrypoint.sh

RUN chmod +x /web/entrypoint.sh

EXPOSE 8000 5432

CMD [ "/web/entrypoint.sh" ]