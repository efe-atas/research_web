FROM nginx:alpine

# Statik dosyaları nginx'e kopyala
COPY index.html /usr/share/nginx/html/
COPY static/ /usr/share/nginx/html/static/

# 80 portunu aç
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

