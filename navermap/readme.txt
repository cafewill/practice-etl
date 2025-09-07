python3 -m playwright install chromium
python3 -m playwright --version

명령 예시 : 
python3 nm-목록수집.py \
  --query "제주도 중문 맛집" \
  --max 150 \
  --out-list "data/list/jeju/*.json" \
  --out-detail "data/detail/jeju/*.json" \
  --show --timeout 42000 --per-item-timeout 4000 \
  --max-images 6 --verbose --screenshot-on-error
