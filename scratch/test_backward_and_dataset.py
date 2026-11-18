import urllib.request
import json

base = 'http://127.0.0.1:8000/api/v1'

print("=== 1. TESTING DATASET EXPLORER ENDPOINTS ===")
# Metadata
req = urllib.request.urlopen(f"{base}/dataset/metadata")
meta = json.loads(req.read().decode('utf-8'))
print(f"[OK] Metadata: {meta['name']}, chosen_sensor: {meta['chosen_sensor']}")

# Summary
req = urllib.request.urlopen(f"{base}/dataset/summary")
summ = json.loads(req.read().decode('utf-8'))
print(f"[OK] Summary: total_records={summ.get('total_records')}, sequence_count={summ.get('sequence_count')}")
print(f"     records_per_activity: {summ.get('records_per_activity')}")
print(f"     activity_distribution: {summ.get('activity_distribution')}")

# Preview (page 1)
req = urllib.request.urlopen(f"{base}/dataset/preview?limit=5&offset=0")
prev = json.loads(req.read().decode('utf-8'))
print(f"[OK] Preview offset 0: total_rows={prev.get('total_rows')}, rows_count={len(prev.get('rows', []))}, records_count={len(prev.get('records', []))}")
print(f"     First row: {prev['rows'][0]}")

# Preview (page 2 pagination test)
req = urllib.request.urlopen(f"{base}/dataset/preview?limit=5&offset=5")
prev2 = json.loads(req.read().decode('utf-8'))
print(f"[OK] Preview offset 5: rows_count={len(prev2.get('rows', []))}, timestamp_s of 1st row: {prev2['rows'][0]['timestamp_s']}")

print("\n=== 2. TESTING BACKWARD ALGORITHM & DUAL VERIFICATION ===")
req = urllib.request.urlopen(f"{base}/dataset/samples")
samples = json.loads(req.read().decode('utf-8'))
print(f"Loaded {len(samples)} sample sequences from dataset.")

for s in samples:
    act = s['activity_name']
    seq = s['raw_sequence']
    payload = json.dumps({'sequence': seq, 'scale_input': True, 'use_scaled': True}).encode('utf-8')
    
    # Run backward
    req_bwd = urllib.request.Request(f"{base}/inference/backward", data=payload, headers={'Content-Type': 'application/json'})
    resp_bwd = json.loads(urllib.request.urlopen(req_bwd).read().decode('utf-8'))
    
    # Run validate
    req_val = urllib.request.Request(f"{base}/inference/validate", data=payload, headers={'Content-Type': 'application/json'})
    resp_val = json.loads(urllib.request.urlopen(req_val).read().decode('utf-8'))
    
    # Check log-space backward as well
    payload_log = json.dumps({'sequence': seq, 'use_scaled': False}).encode('utf-8')
    req_bwd_log = urllib.request.Request(f"{base}/inference/backward", data=payload_log, headers={'Content-Type': 'application/json'})
    resp_bwd_log = json.loads(urllib.request.urlopen(req_bwd_log).read().decode('utf-8'))
    
    print(f"\n--- Activity: {act} (Subject {s['subject_id']}, Length T={s['length']}) ---")
    print(f"  Scaled Backward Log-Likelihood:  {resp_bwd['log_likelihood']:.8f}")
    print(f"  Log-Space Backward Log-Likelihood: {resp_bwd_log['log_likelihood']:.8f}")
    print(f"  Scaled Forward Log-Likelihood:   {resp_val['forward_log_likelihood']:.8f}")
    print(f"  Dual Likelihood Difference (|Δ|): {resp_val['abs_difference_fwd_bwd']:.2e}")
    print(f"  Tolerance (|Δ| < 1e-5) Passed:   {resp_val['tolerance_passed']}")
    print(f"  Beta at t=0:   {[round(x, 6) for x in resp_bwd['scaled_beta_table'][0]]}")
    print(f"  Beta at t=T-1: {[round(x, 6) for x in resp_bwd['scaled_beta_table'][-1]]}")

print("\nALL VERIFICATIONS COMPLETE.")
