ray stop --force
rm -rf /home/tato/MarineEVT/ray_temp/session_*
ray start --head --temp-dir /home/tato/MarineEVT/ray_temp --dashboard-port=8266