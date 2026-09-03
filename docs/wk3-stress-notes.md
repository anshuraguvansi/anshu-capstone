### Finding 1 — Malformed JSON
For all the 4 bad input recieved 422 Unprocessable Content, none of them reached my app code as these validation failed at fastapi layer itself.

### Finding 2 — 5000-character question
Total time took around 4.626 seconds, API didn't hit token limit and streaming worked.

### Finding 3 — Disconnect mid-stream
Sever logged INFO:     127.0.0.1:63720 - "POST /ask HTTP/1.1" 200 OK
Yes server respond to /health after thisas well.

### Finding 4 — 50 parallel requests
Stress test: 50 requests, up to 10 concurrent

Total wall time:   36.90s
Successes:         50 / 50
Effective req/s:   1.36

Latency (successful requests):
  min:   1.21s
  p50:   6.41s
  p95:   11.26s
  max:   14.64s