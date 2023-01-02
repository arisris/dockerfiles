# FROM alpine:3.17.0
FROM frolvlad/alpine-glibc

LABEL maintainer="Aris Riswanto <sksnet.id@gmail.com>"

RUN apk update \
  && apk add micro curl nodejs \
  && apk add php81 php81-pdo php81-pdo_pgsql php81-pdo_mysql php81-pdo_sqlite php81-bcmath php81-ctype php81-fileinfo php81-mbstring php81-bz2 php81-posix php81-dom php81-ftp php81-session php81-tokenizer php81-xml php81-curl php81-brotli \
  && apk add postgresql15 \
  && mkdir /run/postgresql \
  && chown postgres:postgres /run/postgresql/

# Install pnpm static
RUN curl -fsSL "https://github.com/pnpm/pnpm/releases/latest/download/pnpm-linuxstatic-x64" -o /bin/pnpm; chmod +x /bin/pnpm;

RUN addgroup -g 1000 devm \
  && adduser -D -u 1000 -s /bin/sh -h /web -G devm devm \
  && mkdir -p /web/.deno \
  && chown -hR devm:devm /web

COPY ./entrypoint.sh /usr/local/bin/runner
RUN chmod +x /usr/local/bin/runner

USER postgres

RUN mkdir /var/lib/postgresql/data \
  && chmod 0700 /var/lib/postgresql/data \
  && initdb -D /var/lib/postgresql/data \
  && echo "host all all 0.0.0.0/0 md5" >> /var/lib/postgresql/data/pg_hba.conf \
  && echo "listen_addresses='*'" >> /var/lib/postgresql/data/postgresql.conf

USER devm
WORKDIR /web

ENV DENO_INSTALL=/web/.deno
ENV DENO_INSTALL_ROOT=/usr/local/bin

RUN curl -fsSL https://deno.land/x/install/install.sh | sh

EXPOSE 8000 3000 5432

CMD [ "runner" ]