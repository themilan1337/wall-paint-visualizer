#!/bin/bash
set -e

echo "======================================="
echo "🚀 Автоматический деплой Wall Paint API"
echo "======================================="

DOMAIN="api.artegopaints.kz"
EMAIL="ssl@artegopaints.kz"

# 1. Обновление пакетов и установка зависимостей
echo "📦 1/5 Установка Nginx, Certbot и утилит..."
sudo apt-get update -y
sudo apt-get install -y nginx certbot python3-certbot-nginx curl

# 2. Установка Docker (если еще не установлен)
if ! command -v docker &> /dev/null; then
    echo "🐳 2/5 Установка Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    rm get-docker.sh
else
    echo "🐳 2/5 Docker уже установлен, пропускаем..."
fi

# 3. Сборка и запуск проекта в Docker
echo "🏗️ 3/5 Запуск Docker-контейнеров..."
# Используем docker compose (v2) или docker-compose (v1)
if docker compose version &> /dev/null; then
    sudo docker compose up -d --build
else
    sudo docker-compose up -d --build
fi

# 4. Настройка Nginx
echo "⚙️ 4/5 Настройка Nginx в качестве Reverse Proxy..."

# Удаляем дефолтный конфиг Nginx, чтобы не конфликтовал
sudo rm -f /etc/nginx/sites-enabled/default

# Создаем конфигурацию для домена
cat << EOF | sudo tee /etc/nginx/sites-available/$DOMAIN
server {
    listen 80;
    server_name $DOMAIN;

    # Важно: Разрешаем загрузку больших файлов (до 50 МБ), 
    # так как по умолчанию Nginx блокирует файлы больше 1 МБ.
    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Увеличиваем таймауты, так как тяжелые ML-вычисления могут занимать больше минуты
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
EOF

# Включаем сайт
sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/

# Проверяем конфиг и перезапускаем Nginx
sudo nginx -t
sudo systemctl reload nginx

# 5. Получение и установка SSL сертификата
echo "🔒 5/5 Выпуск SSL-сертификата Let's Encrypt..."
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m $EMAIL --redirect

echo "======================================="
echo "✅ Деплой успешно завершен!"
echo "🌐 API доступно по адресу: https://$DOMAIN"
echo "======================================="