#!/bin/bash
set -e

SERVER="root@37.27.200.52"
APP_DIR="/var/www/digitallawyer/app"
LOCAL_DIR="/Users/umyhabiba/Downloads/app 2/"

echo "📦 Syncing updated code to server..."
rsync -avz --progress --delete "$LOCAL_DIR" "$SERVER:$APP_DIR"

ssh $SERVER <<'EOF'
echo "🔧 Fixing ownership & permissions..."
chown -R www-data:www-data /var/www/digitallawyer
chmod 755 /var/www/digitallawyer /var/www/digitallawyer/app

echo "🚫 Killing stray Gunicorn processes..."
pkill -9 -f gunicorn || true
fuser -k 8001/tcp || true

echo "🔄 Restarting Gunicorn & Nginx..."
systemctl daemon-reload
systemctl restart dl-gunicorn
systemctl reload nginx

sleep 5

echo "🩺 Checking local health endpoint..."
curl -s http://127.0.0.1:8001/health || echo "⚠️ Health endpoint failed"

echo "🌍 Checking local homepage (first 20 lines)..."
curl -s http://127.0.0.1:8001/ | head -20
EOF

echo "🌐 Checking public production site..."
curl -s https://digitallawyer.ai | head -20
