#!/bin/bash
# Cookie health check and validation script
# Checks if YouTube cookies are valid and not expired
# Usage: ./scripts/check_cookies.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COOKIES="${PROJECT_DIR}/cookies_netscape.txt"
export PATH="$HOME/.deno/bin:$PATH"

echo "=== YouTube Cookie Health Check ==="
echo "Date: $(date)"
echo "Cookies file: $COOKIES"
echo ""

# Check if cookies file exists
if [ ! -f "$COOKIES" ]; then
    echo "[CRITICAL] Cookies file not found: $COOKIES"
    echo "ACTION: Export cookies from a browser using a cookie exporter extension"
    echo "  Save as Netscape format to: $COOKIES"
    exit 1
fi

# Count cookies
TOTAL_COOKIES=$(awk -F'\t' '/^\.?youtube\.com/ {count++} END {print count+0}' "$COOKIES" 2>/dev/null)
echo "Total YouTube cookies: $TOTAL_COOKIES"

# Check for critical auth cookies
echo ""
echo "=== Critical Auth Cookies ==="
CRITICAL_COOKIES=(
    "__Secure-3PSID"
    "__Secure-3PAPISID" 
    "__Secure-1PSIDTS"
    "__Secure-3PSIDTS"
    "__Secure-YNID"
    "SAPISID"
    "APISID"
    "HSID"
    "SID"
)

MISSING=0
NOW=$(date +%s)
for cookie in "${CRITICAL_COOKIES[@]}"; do
    # Use awk with tab delimiter for exact cookie name matching (field 6)
    LINE=$(awk -F'\t' -v name="$cookie" '$6 == name {print; exit}' "$COOKIES" 2>/dev/null)
    if [ -n "$LINE" ]; then
        EXPIRY=$(echo "$LINE" | awk -F'\t' '{print $5}')
        if [ "$EXPIRY" -gt 0 ] 2>/dev/null && [ "$EXPIRY" -gt "$NOW" ] 2>/dev/null; then
            DAYS_LEFT=$(( (EXPIRY - NOW) / 86400 ))
            echo "  [OK] $cookie - expires in $DAYS_LEFT days"
        elif [ "$EXPIRY" = "0" ] 2>/dev/null; then
            echo "  [OK] $cookie - session cookie"
        else
            echo "  [EXPIRED] $cookie - timestamp $EXPIRY (now: $NOW)"
        fi
    else
        echo "  [MISSING] $cookie"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "=== Test Download ==="

# Try to download a public video to verify cookies work
echo "Testing with public video (dQw4w9WgXcQ)..."
TEST_OUTPUT=$(yt-dlp \
    --cookies "$COOKIES" \
    --remote-components ejs:github \
    --sleep-requests 3 \
    -f "worst[height<=144]" \
    --no-playlist \
    -o "/dev/null" \
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ" 2>&1)

TEST_RET=$?

if [ $TEST_RET -eq 0 ]; then
    echo "  [SUCCESS] Download test passed"
elif echo "$TEST_OUTPUT" | grep -q "Sign in to confirm"; then
    echo "  [FAILED] Cookies are INVALID or EXPIRED - YouTube requires sign-in"
    echo "  ACTION: Export fresh cookies from a browser"
elif echo "$TEST_OUTPUT" | grep -q "rate-limited"; then
    echo "  [WARNING] Rate-limited by YouTube - try again later"
else
    echo "  [WARNING] Download test failed with:"
    echo "  $(echo "$TEST_OUTPUT" | tail -3)"
fi

echo ""
echo "=== Summary ==="
if [ $MISSING -gt 0 ]; then
    echo "[WARNING] $MISSING critical cookie(s) missing"
    echo "ACTION: Manually export fresh cookies from a browser"
    echo "  How to export cookies:"
    echo "  1. Install 'Get cookies.txt LOCALLY' extension in Chrome/Edge"
    echo "  2. Log into YouTube in the browser"
    echo "  3. Click the extension icon and export cookies in Netscape format"
    echo "  4. Copy the exported file to: $COOKIES"
else
    echo "[OK] All critical cookies present"
fi

echo ""
echo "=== Deno JS Runtime ==="
if command -v deno &>/dev/null; then
    echo "[OK] Deno: $(deno --version 2>/dev/null | head -1)"
else
    echo "[MISSING] Deno not found - install with: curl -fsSL https://deno.land/install.sh | sh"
fi

echo ""
echo "=== yt-dlp Version ==="
echo "yt-dlp: $(yt-dlp --version 2>/dev/null)"
