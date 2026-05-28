#!/usr/bin/env bash
# Apply persistent GPU tune for reinstinct on a MI50 (gfx906).
#
# - Lift power-limit fields in pp_table to 300W (workstation VBIOS ships 225W).
# - Raise top mclk (DPM 2+3) from stock 1000 → 1125 MHz.
# - Raise top sclk (DPM 8)   from stock 1725 → 1825 MHz.
# - Pin perflevel high, raise runtime cap to 300W.
#
# pp_table writes are non-persistent (kernel re-loads the in-VBIOS table on
# reboot), so this is normally run via the systemd unit shipped alongside
# (scripts/reinstinct-gpu-tune.service).
#
# Settings landed on after the May-2026 mclk + sclk sweeps:
# linear scaling held all the way to mclk=1150 / sclk=1850 with no errors,
# but we keep one step of margin for long-uptime stability.

set -e
PP_TABLE=/sys/class/drm/card0/device/pp_table

if [[ ! -w "$PP_TABLE" ]]; then
  echo "$PP_TABLE not writable — run as root." >&2
  exit 1
fi

# Wait for amdgpu to settle (we may be invoked very early at boot).
for i in $(seq 1 10); do
  if [[ -r "$PP_TABLE" ]]; then break; fi
  sleep 1
done

upp -p "$PP_TABLE" set --write \
  SmallPowerLimit1=300 \
  SmallPowerLimit2=300 \
  BoostPowerLimit=300 \
  smcPPTable/SocketPowerLimitAc0=300 \
  smcPPTable/SocketPowerLimitDc=300 \
  smcPPTable/FreqTableUclk/2=1125 \
  smcPPTable/FreqTableUclk/3=1125 \
  smcPPTable/FreqTableGfx/8=1825

# Bounce perflevel so the kernel re-reads the table.
rocm-smi --setperflevel auto  >/dev/null
sleep 1
rocm-smi --setperflevel high  >/dev/null
rocm-smi --setpoweroverdrive 300 >/dev/null

echo "[reinstinct-gpu-tune] applied: 300W cap, mclk top=1125 MHz, sclk top=1825 MHz, perflevel=high"
rocm-smi --showclocks    2>&1 | grep -E "sclk|mclk"
rocm-smi --showmaxpower  2>&1 | grep "Max Graphics"
