FROM nginx:1.27-alpine

RUN apk add --no-cache openssl bash

COPY nginx.conf /etc/nginx/nginx.conf
COPY static/ /usr/share/nginx/html/
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /etc/nginx/ssl

EXPOSE 80 443

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
