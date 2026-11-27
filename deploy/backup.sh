#!/bin/sh
# Back up the DataMedic production database and private storage (run on the server from the
# repository root). Keeps the 7 most recent backups. Copy backups off the server as well.
set -eu
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
DEST=${BACKUP_DIR:-./backups}/$STAMP
mkdir -p "$DEST"
COMPOSE="docker compose -f docker-compose.prod.yml --env-file .env.production"
$COMPOSE exec -T db pg_dumpall -U postgres | gzip > "$DEST/postgres.sql.gz"
$COMPOSE run --rm --no-deps -T --entrypoint tar api -C /data -czf - storage > "$DEST/storage.tar.gz"
sha256sum "$DEST"/* > "$DEST/SHA256SUMS"
ls -1dt "${BACKUP_DIR:-./backups}"/*/ | tail -n +8 | xargs -r rm -rf
echo "backup written to $DEST"
