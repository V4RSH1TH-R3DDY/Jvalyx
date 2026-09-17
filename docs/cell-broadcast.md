# Cell Broadcast alerting

A working simulation of how a government emergency alert actually reaches a
phone — wired to Jvalyx's own routing decisions.

## What is real, and what is simulated

The government chain is: an authority authors a **CAP** alert → an aggregator
(India's **SACHET**, built by C-DOT for NDMA) hands it to each carrier's **Cell
Broadcast Centre** → the CBC converts the alert's area into a set of **cell
sites** → those towers broadcast on a one-to-many radio channel (**SIB12** on
LTE/5G, CBCH on GSM) → the phone's **baseband firmware**, which is always
listening, takes over the screen.

| Layer | Here |
|---|---|
| CAP 1.2 authoring | **Real.** `backend/pipeline/cap.py` emits standards-compliant CAP XML. |
| CMAS severity classes | **Real.** 3GPP TS 23.041 message identifiers: 4370 Presidential, 4371 Extreme, 4373 Severe, 4379 Amber. |
| Area-based targeting | **Real in behaviour.** The CBC selects an *area*; devices outside the footprint are never contacted. |
| Radio broadcast | **Simulated.** WebSocket fan-out + ntfy push, because the broadcast channel is licensed spectrum. |
| Baseband takeover | **Simulated.** A web page that goes full-screen with the WEA 853/960 Hz two-tone attention signal. |

The honest limits: a phone has to opt in once by scanning the QR, and unlocated
handsets are judged by the venue's coordinates rather than their own GPS. Real
cell broadcast needs neither, because the receiver ships in the handset by
regulation.

Alerts default to CAP status **`Exercise`** — CAP's own value for a drill. The
handset shows an "Exercise — Simulation" ribbon. Keep it that way in public.

## Demo runbook

**Before the room fills up**

```bash
pip install "qrcode[pil]"                 # optional, for the join QR
export JVALYX_NTFY_TOPIC=jvalyx-fire-xxxx  # optional, locked-screen push
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is the part people forget: bound to localhost, no phone can reach you.

**On stage**

1. Open `http://localhost:8000/cbs/console` on the presenter machine.
2. Audience scans the QR, taps **Attach to network**. Watch the *Attached* counter climb — that tap also unlocks audio, which browsers will not let a page start on its own.
3. Hit **BROADCAST TO FOOTPRINT**. Every handset inside the circle takes over at once.

**The line worth saying:** the receipt reports *cells lit*, not people reached.
The server never learns who is in the room — that is a property of the design,
not a limitation of the demo.

**To fire it from the model instead of a button**

```bash
curl -X POST "http://localhost:8000/cbs/arm?armed=true"
```

Armed, any event crossing into `CRITICAL` broadcasts on its own. It fires on the
*crossing*, so a scenario that sits in CRITICAL does not re-alert every frame.

## Venue Wi-Fi

Client isolation on guest networks blocks phone → laptop, which kills the LAN
path. **Test this in the actual room before presenting.** Fallbacks, in order:

1. Phone hotspot from the presenter's own device, everyone joins that.
2. A tunnel (`ngrok http 8000`) so phones reach you over mobile data.
3. `JVALYX_NTFY_TOPIC` — ntfy rides the OS push channel, so it works over mobile
   data with the screen locked and no tab open. This is the only path that
   survives the venue network failing entirely.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `JVALYX_VENUE_LAT` / `JVALYX_VENUE_LON` | 12.9716 / 77.5946 | Where unlocated handsets are assumed to be. |
| `JVALYX_NTFY_TOPIC` | unset | ntfy topic for locked-screen push. |
| `JVALYX_NTFY_SERVER` | `https://ntfy.sh` | Self-hosted ntfy, if you run one. |
| `JVALYX_PORT` | `8000` | Port used when building the join URL. |

## API

| Endpoint | Purpose |
|---|---|
| `GET /alert` | The handset page. What the QR points at. |
| `GET /cbs/console` | Presenter console: QR, live counts, broadcast trigger. |
| `WS /ws/cbs` | A handset attaching. May send `{latitude, longitude}`. |
| `POST /cbs/broadcast` | Author and broadcast. Takes `event_id`, or explicit coordinates. |
| `GET /cbs/status` | Attached handsets, cells, armed state. |
| `POST /cbs/arm?armed=true` | Auto-broadcast on the CRITICAL crossing. |
| `GET /cbs/alerts/{id}.xml` | The CAP 1.2 XML for a broadcast alert. |
| `GET /cbs/join` · `GET /cbs/qr` | Join URL and its QR image. |

## Why not just push to everyone's IP?

It cannot work, and it is worth being able to say why. Phones do not listen for
unsolicited inbound connections; NAT and firewalls drop it; everyone on one
Wi-Fi shares a single public IP, so IP cannot distinguish people in a room, let
alone locate them. IP geolocation resolves to a city at best and to a carrier
gateway on mobile data.

Cell broadcast solves this by not addressing anyone at all: the tower simply
transmits, and geography is decided by which radio you are in range of. That
inversion is the idea this feature demonstrates.
