#!/usr/bin/env bash
# Apply persistent GPU tune for reinstinct on every MI50/MI60 (gfx906).
#
# - Lift power-limit fields in pp_table to 300W (workstation VBIOS ships 225W).
# - Raise top mclk (DPM 2+3) from stock 1000 → 1125 MHz.
# - Raise top sclk (DPM 8)   from stock 1725 → 1825 MHz.
# - Pin perflevel high, raise runtime cap to 300W.
#
# Cards are found by PCI device id (REINSTINCT_TUNE_DEVICE_IDS, default
# 0x66a0 = MI60, 0x66a1 = MI50 / Radeon Pro VII), not by DRM index — a
# display GPU usually sits at card0. Everything is written through each
# card's own sysfs nodes, so no rocm-smi index mapping is involved.
# REINSTINCT_TUNE_DRY_RUN=1 prints what would be written and changes
# nothing.
#
# pp_table writes are non-persistent (kernel re-loads the in-VBIOS table on
# reboot), so this is normally run via the systemd unit shipped alongside
# (scripts/reinstinct-gpu-tune.service).
#
# Settings landed on after the May-2026 mclk + sclk sweeps:
# linear scaling held all the way to mclk=1150 / sclk=1850 with no errors,
# but we keep one step of margin for long-uptime stability.

set -euo pipefail
IDS="${REINSTINCT_TUNE_DEVICE_IDS:-0x66a0 0x66a1}"
DRY="${REINSTINCT_TUNE_DRY_RUN:-0}"
CAP_W=300

run() { if [[ "$DRY" == 1 ]]; then echo "  would: $*"; else "$@"; fi; }
put() {  # put <value> <sysfs file>
  if [[ "$DRY" == 1 ]]; then echo "  would: echo $1 > $2"; else echo "$1" > "$2"; fi
}

# Wait for amdgpu to settle (we may be invoked very early at boot).
cards=()
for _ in $(seq 1 10); do
  cards=()
  for dev in /sys/class/drm/card[0-9]*/device; do
    [[ -r "$dev/device" && -r "$dev/pp_table" ]] || continue
    id=$(<"$dev/device")
    for want in $IDS; do [[ "$id" == "$want" ]] && cards+=("$(dirname "$dev")"); done
  done
  (( ${#cards[@]} > 0 )) && break
  sleep 1
done
if (( ${#cards[@]} == 0 )); then
  echo "[reinstinct-gpu-tune] no GPU with device id in: $IDS" >&2
  exit 1
fi

for card in "${cards[@]}"; do
  dev="$card/device"
  echo "[reinstinct-gpu-tune] $(basename "$card") ($(basename "$(readlink -f "$dev")"), $(<"$dev/device"))"
  if [[ "$DRY" != 1 && ! -w "$dev/pp_table" ]]; then
    echo "$dev/pp_table not writable — run as root." >&2
    exit 1
  fi
  run upp -p "$dev/pp_table" set --write \
    SmallPowerLimit1=$CAP_W \
    SmallPowerLimit2=$CAP_W \
    BoostPowerLimit=$CAP_W \
    smcPPTable/SocketPowerLimitAc0=$CAP_W \
    smcPPTable/SocketPowerLimitDc=$CAP_W \
    smcPPTable/FreqTableUclk/2=1125 \
    smcPPTable/FreqTableUclk/3=1125 \
    smcPPTable/FreqTableGfx/8=1825
  # Bounce perflevel so the kernel re-reads the table, then pin it high.
  put auto "$dev/power_dpm_force_performance_level"
  [[ "$DRY" == 1 ]] || sleep 1
  put high "$dev/power_dpm_force_performance_level"
  for cap in "$dev"/hwmon/hwmon*/power1_cap; do
    [[ -e "$cap" ]] && put $((CAP_W * 1000000)) "$cap"
  done
done

[[ "$DRY" == 1 ]] && exit 0
echo "[reinstinct-gpu-tune] applied to ${#cards[@]} card(s): ${CAP_W}W cap, mclk top=1125 MHz, sclk top=1825 MHz, perflevel=high"
rocm-smi --showclocks    2>&1 | grep -E "sclk|mclk" || true
rocm-smi --showmaxpower  2>&1 | grep "Max Graphics" || true
