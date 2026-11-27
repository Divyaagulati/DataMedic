#!/usr/bin/env bash
# One-command production setup on a fresh Ubuntu 24.04 VM (x86-64 or arm64), run from the
# repository checkout ON the server, after creating .env.production (mode 600) from
# deploy/.env.production.example:
#
#   ./deploy/server_setup.sh
#
# Idempotent (safe to re-run for updates). Prints progress only, never secret values.
# Steps: swap, Docker Engine, host firewall (80/443), image builds, start the stack (migrations
# run in the `migrate` service), health wait, schema check, nightly backups. The cloud firewall
# (e.g. an Azure NSG or OCI security list) must also allow 22, 80 and 443.
set -euo pipefail
log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }
APP="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP"
C="sudo docker compose -f docker-compose.prod.yml --env-file .env.production"
[ -f .env.production ] || { log "missing .env.production (copy deploy/.env.production.example)"; exit 1; }
chmod 600 .env.production

log "arch: $(uname -m); $(. /etc/os-release && echo "$PRETTY_NAME")"
case "$(uname -m)" in aarch64|x86_64) ;; *) log "unsupported architecture"; exit 1 ;; esac
sudo cloud-init status --wait >/dev/null 2>&1 || true
export DEBIAN_FRONTEND=noninteractive

# 1. Swap (service memory caps total ~7.5 GB; recommended on 4-8 GB machines).
if ! sudo swapon --show | grep -q '^/swapfile'; then
  log "creating 4 GB swap"
  sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

# 2. Docker Engine + compose plugin (official convenience script).
if ! command -v docker >/dev/null 2>&1; then
  log "installing Docker Engine"
  sudo apt-get update -qq
  sudo apt-get install -y -qq ca-certificates curl >/dev/null
  curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
  sudo sh /tmp/get-docker.sh >/tmp/get-docker.log 2>&1
  sudo usermod -aG docker "$USER"
fi
log "$(sudo docker --version); $(sudo docker compose version)"

# 3. Host firewall. Some images (e.g. Oracle's Ubuntu) ship iptables rules that REJECT everything
#    except SSH: open 80/443 before that rule, live and in the persisted rules (edited in place;
#    saving the live table would also persist Docker's chains). No-op where no such rule exists.
for p in 443 80; do
  if ! sudo iptables -C INPUT -p tcp -m state --state NEW -m tcp --dport "$p" -j ACCEPT 2>/dev/null; then
    n=$(sudo iptables -L INPUT --line-numbers -n | awk '$2=="REJECT"{print $1; exit}')
    if [ -n "$n" ]; then
      sudo iptables -I INPUT "$n" -p tcp -m state --state NEW -m tcp --dport "$p" -j ACCEPT
    fi
  fi
done
if [ -f /etc/iptables/rules.v4 ] && ! sudo grep -q -- '--dport 443 -j ACCEPT' /etc/iptables/rules.v4; then
  sudo sed -i '/^-A INPUT -j REJECT/i -A INPUT -p tcp -m state --state NEW -m tcp --dport 80 -j ACCEPT\n-A INPUT -p tcp -m state --state NEW -m tcp --dport 443 -j ACCEPT' /etc/iptables/rules.v4
fi

# 4. Build the two images directly (avoids Compose's bake entitlement prompt), one at a time.
DOMAIN=$(grep -E '^DATAMEDIC_DOMAIN=' .env.production | cut -d= -f2-)
REL=$(grep -E '^DATAMEDIC_RELEASE=' .env.production | cut -d= -f2-)
REL=${REL:-latest}
chmod +x deploy/backup.sh
log "release $REL for https://$DOMAIN"
log "building backend image"
sudo docker build -q -t "datamedic-backend:$REL" backend >/dev/null
log "building web image"
sudo docker build -q -t "datamedic-web:$REL" --build-arg "NEXT_PUBLIC_API_BASE_URL=https://$DOMAIN" apps/web >/dev/null

# 5. Start (migrations run in the 'migrate' service) and wait for the API.
log "starting stack"
$C up -d --no-build
st=""
for _ in $(seq 1 60); do
  st=$($C ps --format '{{.Service}}={{.Health}}' api 2>/dev/null || true)
  [ "$st" = "api=healthy" ] && break
  if [ "$($C ps -a --format '{{.Service}} {{.State}} {{.ExitCode}}' migrate | awk '{print $2, $3}')" = "exited 1" ]; then
    log "migration failed"; $C logs --tail 40 migrate; exit 1
  fi
  sleep 10
done
[ "$st" = "api=healthy" ] || { log "api not healthy"; $C ps; $C logs --tail 40 api; exit 1; }
log "schema: $($C exec -T db psql -U postgres -d datamedic -tAc 'select version_num from alembic_version')"
$C ps --format '{{.Service}}: {{.State}} {{.Health}}'

# 6. Nightly backups (docs/DEPLOYMENT.md section 4), keeps the 7 most recent.
mkdir -p backups
( crontab -l 2>/dev/null | grep -v 'deploy/backup.sh' || true
  echo "0 3 * * * cd $APP && ./deploy/backup.sh >> $HOME/backup.log 2>&1" ) | crontab -
log "nightly backup scheduled (03:00 UTC); done: https://$DOMAIN"
